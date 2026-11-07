"""Exercise the optional native-processing demo without releasing a model."""
import argparse
import hashlib
import json
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--pipeline', default='runs/segment/replay_optimization_v3/runtime_validation_bundle/pipeline.json')
    parser.add_argument('--candidate', default='v2_deployed_guided_verified')
    parser.add_argument('--audit-prefix', default='pipeline_v2_development')
    parser.add_argument('--audit-name', default='2026-11-01_pipeline_v2_ui.json')
    parser.add_argument('--real-photo', help='Existing training photograph for an upload/display smoke check only')
    parser.add_argument('--reference-evaluation', help='Explicit development evaluation directory for runtime class comparison')
    args = parser.parse_args()
    run = ROOT/'runs/segment/replay_optimization_v3'
    manifest_path = ROOT/args.pipeline
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    candidate = ROOT/args.reference_evaluation if args.reference_evaluation else run/'evaluation'/args.candidate
    report = json.loads((candidate/'summary.json').read_text(encoding='utf-8'))
    if 'selection_sha256' in manifest:
        assert report['selection_sha256'] == manifest['selection_sha256'], 'UI reference belongs to another selection'
    record = json.loads((candidate/'records/focused_focused_v2_20261102_val_08_00032.json').read_text(encoding='utf-8'))
    sample = ROOT/'dataset/replay_optimization_v3'/record['image']
    audit = ROOT/'runs/task_audit';errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        try:
            page = browser.new_page(viewport={'width':1440,'height':1100})
            page.on('pageerror', lambda error:errors.append(str(error)))
            page.goto(args.url, wait_until='load')
            expect(page.locator('#mode-badge')).to_contain_text('本机模型已就绪', timeout=60000)
            released = manifest['evaluation'].get('joint_gate_passed') is True and manifest['evaluation'].get('not_deployed') is not True
            expect(page.locator('#active-pipeline')).to_contain_text('通过三项联合验收' if released else '尚未通过识别、误报、轮廓三项联合验收')
            expect(page.locator('#active-pipeline')).not_to_contain_text('NaN')
            expect(page.locator('#start-command')).to_be_visible()
            if released:
                expected = manifest['evaluation']
                expect(page.locator('#metric-primary-value')).to_have_text(str(expected['test_images'])+' 张')
                expect(page.locator('#test-count')).to_have_text(f"{expected['strict_instance_score']*100:.2f}%")
                expect(page.locator('#map50')).to_have_text(str(expected['extra_false_positives']))
                expect(page.locator('#map5095')).to_have_text(f"{expected['boundary_iou']:.4f}")
                expect(page.locator('#metric-scope')).to_contain_text('未见合成图')
                expect(page.locator('#start-command')).to_contain_text('演示首版对照.cmd')
                expect(page.locator('#start-command')).to_contain_text('选择图片')
            else:
                expect(page.locator('#metric-primary-value')).to_have_text('9 类')
                expect(page.locator('#metric-scope')).to_contain_text('首版合成测试集')
            expect(page.locator('#model-mask-option')).to_be_hidden()
            health = page.request.get(args.url+'/api/health').json()
            assert health['model_sha256'] == report['provenance']['model_sha256']
            assert health['pipeline_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            page.get_by_label('选择图片', exact=True).set_input_files(str(sample))
            with page.expect_response(lambda r:'/api/predict' in r.url, timeout=120000) as response:
                page.get_by_role('button', name='识别形状', exact=True).click()
            assert response.value.status == 200, response.value.status
            # Large PNG previews can exceed Chromium's inspector response cache.
            # Verify the result that the application actually consumed/rendered.
            page.wait_for_function("uploaded && uploaded.status === 'done'", timeout=60000)
            value = page.evaluate("({predictions:uploaded.predictions, settings:uploaded.settings, model_sha256:uploaded.model_sha256})")
            expect(page.locator('#case-status')).to_contain_text('按照片边缘校正的轮廓')
            expect(page.locator('#model-mask-option')).to_be_visible()
            predictions = value['predictions']
            assert [r['class_id'] for r in predictions] == [r['class_id'] for r in record['predictions']]
            if manifest['version'] in {4, 5}:
                assert value['settings']['component_image_sizes'] == [320,640,320]
                expect(page.locator('#case-status')).to_contain_text('组合检测尺寸 320 / 640 / 320')
            paths = page.locator('#prediction-overlay path[data-mask-native]')
            expect(paths).to_have_count(len(predictions))
            guided_paths = paths.evaluate_all('(nodes) => nodes.map(node => node.getAttribute("d"))')
            expected_paths = page.evaluate('''items => {
              const svg=document.getElementById('prediction-overlay'),w=svg.viewBox.baseVal.width,h=svg.viewBox.baseVal.height;
              return items.map(item=>item.model_mask_contours.filter(r=>r.length).map(r=>r.map(([x,y],i)=>(i?'L':'M')+(x*w)+' '+(y*h)).join(' ')+' Z').join(' '));
            }''', predictions)
            page.locator('#show-model-mask').check()
            expect(page.locator('#case-status')).to_contain_text('模型原始轮廓')
            original_paths = paths.evaluate_all('(nodes) => nodes.map(node => node.getAttribute("d"))')
            assert original_paths == expected_paths
            assert original_paths != guided_paths
            page.locator('#show-model-mask').uncheck()
            assert paths.evaluate_all('(nodes) => nodes.map(node => node.getAttribute("d"))') == guided_paths
            page.locator('#show-box').uncheck()
            page.locator('#show-mask').uncheck()
            expect(paths).to_have_count(0)
            page.locator('#show-mask').check()
            expect(paths).to_have_count(len(predictions))
            page.screenshot(path=str(audit/(args.audit_prefix+'_desktop.png')), full_page=True)
            page.locator('.case-button').first.click()
            expect(page.locator('#model-mask-option')).to_be_hidden()
            page.locator('#show-upload').click()
            expect(page.locator('#show-model-mask')).not_to_be_checked()
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            page.screenshot(path=str(audit/(args.audit_prefix+'_mobile.png')), full_page=True)
            page.locator('#confidence').fill('0.35')
            page.locator('#confidence').dispatch_event('input')
            expect(page.locator('#active-pipeline')).to_contain_text('当前阈值已改变')
            expect(page.locator('#model-mask-option')).to_be_hidden()
            if released:
                expect(page.locator('#test-count')).to_have_text(f"{expected['strict_instance_score']*100:.2f}%")
            real_smoke = None
            if args.real_photo:
                real_path = ROOT/args.real_photo
                page.set_viewport_size({'width':1440,'height':1100})
                page.locator('#confidence').fill(str(value['settings']['conf']))
                page.locator('#confidence').dispatch_event('input')
                page.get_by_label('选择图片', exact=True).set_input_files(str(real_path))
                with page.expect_response(lambda r:'/api/predict' in r.url, timeout=180000) as real_response:
                    page.get_by_role('button', name='识别形状', exact=True).click()
                assert real_response.value.status == 200
                page.wait_for_function("uploaded && uploaded.status === 'done'", timeout=60000)
                real_value = page.evaluate("({predictions:uploaded.predictions,width:uploaded.width,height:uploaded.height,elapsed_seconds:uploaded.elapsed_seconds})")
                assert real_value['predictions'], 'The known positive photograph returned no detections'
                assert all(0 <= item['class_id'] < 9 and 0 <= item['confidence'] <= 1 for item in real_value['predictions'])
                expect(page.locator('#prediction-overlay path[data-mask-native]')).to_have_count(len(real_value['predictions']))
                expect(page.locator('#model-mask-option')).to_be_visible()
                page.locator('#show-model-mask').check()
                expect(page.locator('#case-status')).to_contain_text('模型原始轮廓')
                page.locator('#show-model-mask').uncheck()
                expect(page.locator('#case-status')).to_contain_text('按照片边缘校正的轮廓')
                page.screenshot(path=str(audit/(args.audit_prefix+'_real_training_photo.png')), full_page=True)
                real_smoke = {'image':real_path.relative_to(ROOT).as_posix(),
                              'image_sha256':hashlib.sha256(real_path.read_bytes()).hexdigest(),
                              'detections':len(real_value['predictions']),'elapsed_seconds':real_value['elapsed_seconds'],
                              'purpose':'existing training photograph; upload/display check, not accuracy evaluation',
                              'independent_real_test':False}
            assert not errors, errors
            result = {'passed':True,'development_runtime_only':not released,'release_qualified':released,
                      'default_deployment_verified':False,
                      'checks':['real_upload','class_consistency','original_guided_toggle','native_evenodd_paths',
                                'mask_and_box_toggles','historical_examples','mobile_no_overflow','confidence_notice',
                                'accepted_metrics' if released else 'development_metrics','startup_instructions'],
                      'pipeline_sha256':health['pipeline_sha256'],'detections':len(predictions),
                      'real_training_photo_smoke':real_smoke,'javascript_errors':errors}
            if real_smoke:
                result['checks'].append('real_training_photo_upload_and_contour_toggle')
            (audit/args.audit_name).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(result,ensure_ascii=False))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
