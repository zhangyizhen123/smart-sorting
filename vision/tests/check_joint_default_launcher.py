"""Verify the activated Windows entrypoint loads the accepted pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'models/trained/stl_shape_joint_v4'
AUDIT = ROOT / 'runs/task_audit'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def listening(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(.5)
        return connection.connect_ex(('127.0.0.1', port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9881)
    parser.add_argument('--audit-name', default='2026-11-01_joint_round4_default_launcher.json')
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This check exercises the Windows .cmd entrypoint')
    if listening(args.port):
        raise RuntimeError('The isolated verification port is already occupied')
    activation = read(RELEASE / 'activation.json')
    manifest = RELEASE / 'pipeline.json'
    launcher = ROOT / '演示模型.cmd'
    assert digest(manifest) == activation['pipeline_sha256']
    assert digest(launcher) == activation['launcher_sha256']
    assert digest(ROOT / '演示首版对照.cmd') == activation['original_launcher_sha256']
    assert digest(ROOT / activation['rollback']) == activation['original_launcher_sha256']
    focused = ROOT / '演示专项模型.cmd'
    focused_before = digest(focused)
    candidate = read(ROOT / 'runs/segment/joint_round_v4/sealed_test/candidate/summary.json')
    released = read(manifest)
    assert candidate['provenance']['pipeline_sha256'] == released['release_origin']['source_pipeline_sha256']
    base_reference = released['sources']['base']
    base_path = manifest.parent / base_reference['pipeline']
    assert digest(base_path) == base_reference['pipeline_sha256']
    primary_reference = read(base_path)['sources']['primary']
    primary_path = base_path.parent / primary_reference['pipeline']
    assert digest(primary_path) == primary_reference['pipeline_sha256']
    primary = read(primary_path)
    expected_model = primary['model_sha256']
    assert digest(primary_path.parent / primary['model']) == expected_model
    AUDIT.mkdir(parents=True, exist_ok=True)
    log_path = AUDIT / (Path(args.audit_name).stem + '.log')
    result = {'passed': False, 'default_deployment_verified': False,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'launcher': launcher.relative_to(ROOT).as_posix(),
              'launcher_sha256': digest(launcher), 'port': args.port,
              'explicit_pipeline_argument': False,
              'independent_real_test': False}
    try:
        with tempfile.TemporaryDirectory(prefix='sorting-launch-check-') as directory:
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(
                    [os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', str(launcher),
                     '--port', str(args.port), '--no-open'],
                    cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                result['launcher_pid'] = process.pid
                try:
                    deadline = time.monotonic() + 120
                    health = None
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError('Default launcher exited; inspect the verification log')
                        try:
                            with urllib.request.urlopen(
                                    f'http://127.0.0.1:{args.port}/api/health', timeout=2) as response:
                                health = json.load(response)
                            break
                        except (urllib.error.URLError, TimeoutError):
                            time.sleep(.5)
                    if health is None:
                        raise RuntimeError('Default launcher did not become ready within 120 seconds')
                    assert health['ready'] is True
                    assert health['pipeline_sha256'] == activation['pipeline_sha256']
                    assert health['model_sha256'] == expected_model
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/', timeout=5) as response:
                        html = response.read().decode('utf-8')
                    assert '九类形状模型 · 图片识别与轮廓查看' in html
                    assert digest(focused) == focused_before
                    result.update(health=health, started_outside_project=True,
                                  focused_launcher_sha256=focused_before,
                                  original_launcher_preserved=True,
                                  checks=['actual_cmd_entrypoint', 'no_pipeline_override',
                                          'release_and_model_hashes', 'page_served',
                                          'original_and_focused_launchers_preserved'])
                finally:
                    if process.poll() is None:
                        cleanup = subprocess.run(
                            ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
                            timeout=15)
                        result['cleanup_exit_code'] = cleanup.returncode
                        process.wait(timeout=15)
                    deadline = time.monotonic() + 10
                    while listening(args.port) and time.monotonic() < deadline:
                        time.sleep(.2)
                    result['verification_port_closed'] = not listening(args.port)
                    if not result['verification_port_closed']:
                        raise RuntimeError('Verification server remains on the isolated port')
        result.update(passed=True, default_deployment_verified=True)
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        (AUDIT / args.audit_name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
