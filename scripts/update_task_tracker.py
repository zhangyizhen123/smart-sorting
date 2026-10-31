"""Maintain task data and synchronized Markdown / formatted Excel views."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
import io
from pathlib import Path
import re

from task_tracker_excel import BOOK, build_book, check_book

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/任务跟踪表.csv'
VIEW = ROOT / 'docs/任务跟踪表.md'
LOG = ROOT / 'docs/任务更新记录.md'
FIELDS = ['编号', '模块', '任务', '状态', '优先级', '独立推进条件', '现状或证据',
          '需要的信息或数据', '我可以完全负责的部分', '需要你或队友完成的部分',
          '依赖任务', '完成标准', '下一步', '最近更新', '完成日期', '备注']
STATUSES = ('已确认', '已完成', '部分完成', '待开始', '待资料', '待决策', '待实测', '暂缓')
CLOSED = {'已确认', '已完成'}


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('CSV 列名或顺序不匹配')
        rows = list(reader)
    validate(rows)
    return rows


def validate(rows: list[dict[str, str]]) -> None:
    ids = [row['编号'] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('任务为空或编号重复')
    dependencies: dict[str, list[str]] = {}
    for row in rows:
        if set(row) != set(FIELDS) or any(value is None for value in row.values()):
            raise ValueError('CSV 行的字段数量不匹配')
        task_id = row['编号']
        if not re.fullmatch(r'[A-Z]+-\d{2}', task_id):
            raise ValueError(f'非法编号：{task_id}')
        if row['状态'] not in STATUSES or row['优先级'] not in ('P0', 'P1', 'P2'):
            raise ValueError(f'{task_id} 的状态或优先级不合法')
        for key in FIELDS:
            if key != '完成日期' and not row[key].strip():
                raise ValueError(f'{task_id} 缺少 {key}')
        for key in ('最近更新', '完成日期'):
            if row[key]:
                date.fromisoformat(row[key])
        if row['状态'] not in CLOSED and row['完成日期']:
            raise ValueError(f'{task_id} 尚未完成却填了完成日期')
        dep_ids = [] if row['依赖任务'] == '无' else row['依赖任务'].split(';')
        if any(dep not in ids or dep == task_id for dep in dep_ids):
            raise ValueError(f'{task_id} 的依赖编号不存在或依赖自身')
        dependencies[task_id] = dep_ids
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f'循环依赖：{task_id}')
        if task_id in visited:
            return
        visiting.add(task_id)
        for dep in dependencies[task_id]:
            visit(dep)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        visit(task_id)


def cell(value: str) -> str:
    return value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')


def table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    return ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] + [
        '| ' + ' | '.join(cell(row[column]) for column in columns) + ' |' for row in rows
    ]


def render(rows: list[dict[str, str]]) -> str:
    counts = Counter(row['状态'] for row in rows)
    pending = [row for row in rows if row['状态'] not in CLOSED]
    lines = ['# 智能分拣任务跟踪表', '',
             f"最近更新：{max(row['最近更新'] for row in rows)}。共 {len(rows)} 项；已完成 {counts['已完成']} 项、已确认 {counts['已确认']} 项、未完成 {len(pending)} 项（含部分完成和暂缓）。", '',
             '正式阅读版：[任务跟踪表.xlsx](任务跟踪表.xlsx)（分工作表、清晰表头、换行、列宽、筛选和冻结窗格）。维护数据：[任务跟踪表.csv](任务跟踪表.csv)；[更新记录](任务更新记录.md)。', '',
             '范围：资料整理、已确认装置方案、历史实验、视觉软件、标定与数据、UART、STM32、转台、光电、显示屏及整机验收。', '',
             '**完成口径**：已确认只代表你已选定方案；已完成只针对该行定义的产物/验收范围。代码骨架和桌面测试不代表板级部署或实机验证。历史完成日期不明时留空；最近更新是核对日期。任务颗粒度不同，不以条目比例代表整机工程完成率。', '',
             '**独立推进条件**：现在可独立＝现有资料足够完成该行的软件或文档工作；资料齐后可独立＝需先提供输入；需线下配合＝我可全责软件/分析部分，实物操作仍需你们；需你决定＝先确认产品行为；已闭环＝该行范围完成。', '',
             '**优先级**：P0 阻塞联调或影响动作正确性；P1 主流程必需；P2 整理、增强与后续完善。', '',
             '**协作约定**：每完成并验证一项立即更新状态、证据和日期；用户新决定优先。对话和现有文件无法证明的实物进度，一律不推定完成。你或队友线下完成后提供结果即可更新。', '',
             '## 状态总览', '', '| 状态 | 数量 |', '| --- | ---: |']
    lines.extend(f'| {state} | {counts[state]} |' for state in STATUSES)
    lines += ['', '## 待补资料与待定事项索引', '',
              '下面只列需外部资料、决策或实物验证的入口；每项的具体分工和验收标准在后面的详细表中。', '']
    inputs = [row for row in pending if row['独立推进条件'] != '现在可独立']
    lines += table(inputs, ['编号', '任务', '需要的信息或数据'])
    lines += ['', '## 可立即由我独立推进', '']
    independent = [row for row in pending if row['独立推进条件'] == '现在可独立']
    lines += table(independent, ['编号', '优先级', '任务', '我可以完全负责的部分', '依赖任务'])
    lines += ['', '## 已完成与已确认', '']
    for module in dict.fromkeys(row['模块'] for row in rows):
        selected = [row for row in rows if row['模块'] == module and row['状态'] in CLOSED]
        if selected:
            lines += [f'### {module}', ''] + table(selected, ['编号', '任务', '状态', '现状或证据', '完成标准', '备注']) + ['']
    lines += ['## 未完成任务明细', '']
    for module in dict.fromkeys(row['模块'] for row in rows):
        selected = [row for row in pending if row['模块'] == module]
        if selected:
            lines += [f'### {module}', ''] + table(selected, ['编号', '任务', '状态', '优先级', '独立推进条件', '现状或证据', '需要的信息或数据', '我可以完全负责的部分', '需要你或队友完成的部分', '依赖任务', '完成标准', '下一步', '备注']) + ['']
    lines += ['## 更新方法', '',
              '在项目根目录执行；完成任务时必须写实际证据，不以“准备完成”或仅修改状态代替验收。', '',
              '```powershell', 'python scripts/update_task_tracker.py update 任务编号 --status 已完成 --evidence "产物路径与验证结果" --note "本次完成说明"',
              'python scripts/update_task_tracker.py render', 'python scripts/update_task_tracker.py check', '```', '',
              'CSV 直接编辑后执行 render。程序 update 会同步 CSV、Markdown 和 Excel，并追加更新记录。Excel 是生成的阅读版；若直接修改了工作簿，请先合并回维护数据再重新生成。', '']
    return '\n'.join(lines)


def write_csv(rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    temporary = SOURCE.with_name(SOURCE.name + '.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8-sig', newline='')
    temporary.replace(SOURCE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('render')
    subparsers.add_parser('check')
    update = subparsers.add_parser('update')
    update.add_argument('task_id')
    update.add_argument('--status', choices=STATUSES, required=True)
    update.add_argument('--evidence', required=True)
    update.add_argument('--note', required=True)
    update.add_argument('--date', default=date.today().isoformat())
    args = parser.parse_args()
    rows = read_rows()
    if args.command == 'update':
        task = next((row for row in rows if row['编号'] == args.task_id), None)
        if task is None:
            parser.error('任务编号不存在')
        old = task['状态']
        day = date.fromisoformat(args.date).isoformat()
        if not args.evidence.strip() or not args.note.strip():
            parser.error('证据和变更说明不能为空')
        if old == args.status and task['现状或证据'] == args.evidence:
            parser.error('状态与证据没有变化；不写入空更新')
        task.update({'状态': args.status, '现状或证据': args.evidence, '最近更新': day})
        if args.status in CLOSED:
            task['完成日期'] = task['完成日期'] or day
            task['独立推进条件'] = '已闭环'
            task['下一步'] = '保持回归；有新变更时重新核验'
        else:
            task['完成日期'] = ''
            if task['独立推进条件'] == '已闭环':
                task['独立推进条件'] = '需你决定'
                task['下一步'] = '根据重新打开原因更新依赖、分工和所缺信息'
        task['备注'] = args.note
        validate(rows)
        write_csv(rows)
        with LOG.open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {day} | {task["编号"]} | {old} → {args.status} | {args.note} | 证据：{args.evidence}\n')
    content = render(rows)
    if args.command == 'check':
        if not VIEW.exists() or VIEW.read_text(encoding='utf-8') != content:
            parser.error('Markdown 与 CSV 不一致；请先执行 render')
        check_book(rows)
        print(f'OK: {len(rows)} tasks; unique IDs; valid dependencies; Markdown and Excel synchronized')
    else:
        build_book(rows)
        VIEW.write_text(content, encoding='utf-8', newline='\n')
        print(f'Updated {len(rows)} tasks: {VIEW}; {BOOK}')


if __name__ == '__main__':
    main()
