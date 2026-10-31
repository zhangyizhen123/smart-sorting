"""Generate a readable, multi-sheet task tracker from the canonical CSV rows."""

from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import unicodedata

import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'docs/任务跟踪表.xlsx'
RENDER_VERSION = '1'
CLOSED = {'已确认', '已完成'}
STATUSES = ('已完成', '已确认', '部分完成', '待开始', '待资料', '待决策', '待实测', '暂缓')
COLORS = {
    '已完成': ('#DCF1E5', '#19643A'), '已确认': ('#E4EFFA', '#285783'),
    '部分完成': ('#FFF0CB', '#815B0A'), '待开始': ('#EDF0F4', '#4E5868'),
    '待资料': ('#FFE6CE', '#8A4B15'), '待决策': ('#EEE5F7', '#704692'),
    '待实测': ('#DDF0F2', '#27656D'), '暂缓': ('#E8E8E8', '#666666'),
}
DEFINITIONS = {
    '编号': '稳定的任务编号；点击可跳转到关联明细。',
    '模块': '任务所属的项目模块。',
    '任务': '本项工作的具体内容和边界。',
    '状态': '已完成仅针对本项验收范围；已确认只是方案决定。',
    '优先级': 'P0：阻塞联调或影响动作正确性；P1：主流程必需；P2：整理或增强。',
    '独立推进条件': '现在可独立 / 资料齐后可独立 / 需线下配合 / 需你决定。',
    '现状或证据': '当前已有产物、检查结果或未完成的具体缺口。',
    '需要的信息或数据': '启动或完成任务前，需要你提供或现场采集的资料。',
    '我可以完全负责的部分': '助手可全责完成的软件、分析、文档或数据处理范围。',
    '需要你或队友完成的部分': '需要现场接线、测量、拍摄、验证或由你做出的决定。',
    '依赖任务': '建议先完成的任务编号；以分号分隔。',
    '完成标准': '标记本项完成之前应满足的可核验条件。',
    '下一步': '下一项具体行动。',
    '最近更新': '本条状态或证据最后一次核对日期，不等于最初完成日期。',
    '完成日期': '有证据可确定的完成日期；历史日期不明时留空。',
    '备注': '范围限制、用户决定或本次变更说明。',
}
SHEETS = [
    ('未完成任务', 'pending', [
        ('编号', '任务编号', 10), ('模块', '所属模块', 20), ('任务', '任务内容', 36),
        ('状态', '当前状态', 12), ('优先级', '优先级', 10),
        ('独立推进条件', '推进条件', 19), ('下一步', '下一步行动', 48),
    ], '主要查看进度；点击任务编号查看资料与分工。筛选按钮可按状态、模块或优先级查找。'),
    ('资料与分工', 'pending', [
        ('编号', '任务编号', 10), ('任务', '任务内容', 34),
        ('需要的信息或数据', '需要的信息 / 数据', 53),
        ('我可以完全负责的部分', '我可以完全负责的部分', 49),
        ('需要你或队友完成的部分', '需要你 / 队友完成的部分', 43),
    ], '只列未完成任务；第三列写缺什么，第四列写我负责什么，第五列写你们需要做什么。'),
    ('验收标准', 'all', [
        ('编号', '任务编号', 10), ('任务', '任务内容', 36), ('状态', '当前状态', 12),
        ('依赖任务', '前置任务编号', 24), ('完成标准', '怎样才算完成', 65),
    ], '覆盖全部任务；用“前置任务编号”了解依赖，用最后一列确认完成标准。'),
    ('证据与备注', 'all', [
        ('编号', '任务编号', 10), ('任务', '任务内容', 34), ('状态', '当前状态', 12),
        ('现状或证据', '当前进展 / 验证证据', 60), ('备注', '说明与备注', 45),
        ('最近更新', '最近更新日期', 14), ('完成日期', '确认完成日期', 14),
    ], '覆盖全部任务；证据和日期在这里查。历史完成日期不明的格子留空，没有补造日期。'),
    ('已完成与确认', 'closed', [
        ('编号', '任务编号', 10), ('模块', '所属模块', 20), ('任务', '任务内容', 38),
        ('状态', '当前状态', 12), ('现状或证据', '完成产物 / 确认依据', 63),
        ('完成日期', '确认完成日期', 14),
    ], '“已确认”代表方案已选定；“已完成”只针对该项范围，不等同于整机实测完成。'),
]


def fingerprint(rows: list[dict[str, str]]) -> str:
    payload = json.dumps({'renderer': RENDER_VERSION, 'rows': rows}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def text_lines(value: str, width: float) -> int:
    """Conservative visual wrapping estimate: Chinese glyphs occupy two units."""
    capacity = max(6, int((width - 2) * 0.86))
    lines = 0
    for paragraph in str(value).split('\n'):
        current = 0
        lines += 1
        for char in paragraph:
            units = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
            if current + units > capacity:
                lines += 1
                current = 0
            current += units
    return lines


def formats(workbook: xlsxwriter.Workbook) -> dict:
    base = {'font_name': 'Microsoft YaHei', 'font_size': 11, 'font_color': '#263445'}
    result = {
        'title': workbook.add_format({**base, 'font_size': 18, 'bold': True, 'font_color': 'white', 'bg_color': '#223750', 'valign': 'vcenter'}),
        'subtitle': workbook.add_format({**base, 'font_size': 10, 'font_color': '#53677B', 'text_wrap': True, 'valign': 'vcenter'}),
        'header': workbook.add_format({**base, 'bold': True, 'font_color': 'white', 'bg_color': '#314B68', 'text_wrap': True, 'valign': 'vcenter', 'bottom': 2, 'bottom_color': '#58A9AE'}),
        'body': workbook.add_format({**base, 'text_wrap': True, 'valign': 'top'}),
        'link': workbook.add_format({**base, 'font_color': '#146B96', 'underline': 1, 'text_wrap': True, 'valign': 'top'}),
        'date': workbook.add_format({**base, 'num_format': 'yyyy-mm-dd', 'align': 'center', 'valign': 'top'}),
        'center': workbook.add_format({**base, 'align': 'center', 'valign': 'top', 'text_wrap': True}),
        'note': workbook.add_format({**base, 'bg_color': '#EDF4F7', 'text_wrap': True, 'valign': 'vcenter'}),
        'section': workbook.add_format({**base, 'bold': True, 'font_size': 12, 'bg_color': '#E3EBF2', 'valign': 'vcenter'}),
        'number': workbook.add_format({**base, 'bold': True, 'font_size': 15, 'align': 'center', 'num_format': '0'}),
        'count': workbook.add_format({**base, 'num_format': '0', 'align': 'center'}),
    }
    result['status'] = {status: workbook.add_format({**base, 'bg_color': bg, 'font_color': fg, 'bold': True, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True}) for status, (bg, fg) in COLORS.items()}
    result['priority'] = {key: workbook.add_format({**base, 'font_color': color, 'bold': True, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True}) for key, color in [('P0', '#AA3030'), ('P1', '#865700'), ('P2', '#3E627E')]}
    return result


def configure_sheet(sheet, zoom: int = 90) -> None:
    sheet.hide_gridlines(2)
    sheet.set_zoom(zoom)
    sheet.set_landscape()
    sheet.set_paper(9)
    sheet.fit_to_pages(1, 0)
    sheet.set_margins(0.25, 0.25, 0.4, 0.4)
    sheet.set_footer('&L智能分拣 · 任务跟踪&R第 &P 页 / 共 &N 页')


def build_book(rows: list[dict[str, str]], destination: Path = BOOK) -> None:
    """Write a fresh workbook without touching an already-open user workbook."""
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + '.tmp.xlsx')
    pending = [row for row in rows if row['状态'] not in CLOSED]
    closed = [row for row in rows if row['状态'] in CLOSED]
    selections = {'pending': pending, 'closed': closed, 'all': rows}
    locations = {name: {row['编号']: index + 5 for index, row in enumerate(selections[scope])} for name, scope, _, _ in SHEETS}
    latest = max(row['最近更新'] for row in rows)
    counts = Counter(row['状态'] for row in rows)
    with xlsxwriter.Workbook(temporary, {'strings_to_formulas': False, 'strings_to_urls': False}) as book:
        book.set_properties({'title': '智能分拣任务跟踪表', 'subject': '进度、所需资料、分工与验收', 'comments': '由任务 CSV 同步生成的正式 Excel 阅读版'})
        book.set_custom_property('TaskTrackerSourceSHA256', fingerprint(rows))
        fmt = formats(book)
        overview = book.add_worksheet('总览与使用说明')
        configure_sheet(overview, 100)
        overview.set_tab_color('#223750')
        overview.set_column('A:A', 11)
        overview.set_column('B:B', 25)
        overview.set_column('C:D', 12)
        overview.set_column('E:E', 26)
        overview.set_column('F:F', 26)
        overview.set_column('G:G', 22)
        overview.merge_range('A1:G1', '智能分拣 · 任务跟踪表', fmt['title'])
        overview.set_row(0, 38)
        overview.merge_range('A2:G2', f'最近更新：{latest}   ·   {len(rows)} 项任务   ·   已完成 {counts["已完成"]} 项   ·   已确认 {counts["已确认"]} 项   ·   未完成 {len(pending)} 项', fmt['subtitle'])
        overview.set_row(1, 28)
        overview.merge_range('A3:G3', '从下方入口选择要看的内容。每页只有 5–7 列；顶部是中文表头，长文字自动换行。', fmt['note'])
        overview.set_row(2, 30)
        overview.merge_range('A5:C5', '进度概况', fmt['section'])
        overview.merge_range('E5:G5', '点击进入任务页面', fmt['section'])
        overview.write_row('A6', ['状态', '条目数'], fmt['header'])
        overview.merge_range('C6:D6', '含义', fmt['header'])
        overview.set_row(5, 30)
        meanings = {'已完成': '该项范围完成', '已确认': '方案已选定', '部分完成': '仍需补齐', '待开始': '尚未实施', '待资料': '缺少输入', '待决策': '等你决定', '待实测': '需要现场验证', '暂缓': '暂不推进'}
        for index, status in enumerate(STATUSES, start=6):
            overview.write(index, 0, status, fmt['status'][status])
            formula = f'=COUNTIF(\'未完成任务\'!D5:D{len(pending)+4},A{index+1})+COUNTIF(\'已完成与确认\'!D5:D{len(closed)+4},A{index+1})'
            overview.write_formula(index, 1, formula, fmt['number'], counts[status])
            overview.merge_range(index, 2, index, 3, meanings[status], fmt['body'])
            overview.set_row(index, 30)
        descriptions = {
            '未完成任务': '状态、优先级和下一步', '资料与分工': '缺什么、我负责什么、你们做什么',
            '验收标准': '先做哪些任务、怎样才算完成', '证据与备注': '已有结果、说明和更新日期',
            '已完成与确认': '查看已完成产物与已定方案',
        }
        for index, (name, _, _, _) in enumerate(SHEETS, start=6):
            overview.write_url(index, 4, f"internal:'{name}'!A1", fmt['link'], name)
            overview.merge_range(index, 5, index, 6, descriptions[name], fmt['body'])
        overview.merge_range('E13:G15', '阅读建议：先看“未完成任务”；需要准备材料时看“资料与分工”。\n每页第 4 行是表头，筛选可按任务编号快速定位。', fmt['note'])
        overview.merge_range('A17:G17', '如何理解这些状态', fmt['section'])
        overview.merge_range('A18:G19', '“已确认”不代表代码或硬件已完成。软件骨架、桌面测试、板级部署和实机验证分别记账；不要用任务数量比例推断整机完成率。', fmt['note'])
        overview.set_row(17, 24)
        overview.set_row(18, 24)
        overview.merge_range('A21:G21', '模块分布', fmt['section'])
        module_headers = ['模块', '全部任务', '已完成', '已确认', '未完成']
        overview.merge_range('A22:C22', module_headers[0], fmt['header'])
        overview.write_row('D22', module_headers[1:], fmt['header'])
        for index, module in enumerate(dict.fromkeys(row['模块'] for row in rows), start=22):
            group = [row for row in rows if row['模块'] == module]
            module_counts = Counter(row['状态'] for row in group)
            overview.merge_range(index, 0, index, 2, module, fmt['body'])
            overview.write_row(index, 3, [len(group), module_counts['已完成'], module_counts['已确认'], sum(row['状态'] not in CLOSED for row in group)], fmt['count'])
            overview.set_row(index, 25)
        overview.merge_range('A36:G36', '优先级与推进条件', fmt['section'])
        overview.merge_range('A37:G38', 'P0：阻塞联调或影响动作正确性；P1：主流程必需；P2：整理或增强。\n“现在可独立”表示我可以开始软件/文档工作；有前置任务时仍须按依赖顺序推进。', fmt['note'])
        overview.set_row(36, 25)
        overview.set_row(37, 25)
        overview.merge_range('A40:G40', '后续怎样更新', fmt['section'])
        overview.merge_range('A41:G43', '每完成并验证一项，我会同步更新 Excel、Markdown 和任务数据，并记录证据、日期。线下硬件进展需你们反馈。\n本文件是同步生成的阅读版；如直接在 Excel 中改了内容，请告知我合并回台账，避免下次生成覆盖修改。历史完成日期不明的格子留空。', fmt['note'])
        for index in range(40, 43):
            overview.set_row(index, 26)
        overview.freeze_panes(3, 0)
        overview.print_area('A1:G43')
        overview.activate()
        overview.set_selection('A1')

        for table_index, (name, scope, columns, instruction) in enumerate(SHEETS, start=1):
            selected = selections[scope]
            sheet = book.add_worksheet(name)
            configure_sheet(sheet, 90 if len(columns) < 7 else 85)
            sheet.set_tab_color('#419879' if scope == 'closed' else '#468A9A' if scope == 'pending' else '#748AA4')
            last_col = len(columns) - 1
            sheet.merge_range(0, 0, 0, last_col-1, f'{name}  ·  {len(selected)} 项', fmt['title'])
            sheet.write_url(0, last_col, "internal:'总览与使用说明'!A1", fmt['link'], '返回总览')
            sheet.set_row(0, 36)
            sheet.merge_range(1, 0, 1, last_col, instruction, fmt['subtitle'])
            sheet.set_row(1, 32)
            sheet.set_row(2, 8)
            sheet.set_row(3, 36)
            for col, (field, label, width) in enumerate(columns):
                sheet.set_column(col, col, width, fmt['body'])
                sheet.write_comment(3, col, DEFINITIONS[field], {'author': '字段说明', 'width': 300, 'height': 85})
            sheet.add_table(3, 0, 3+len(selected), last_col, {
                'name': f'TaskView{table_index}', 'style': 'Table Style Medium 2', 'autofilter': True,
                'columns': [{'header': label, 'header_format': fmt['header']} for _, label, _ in columns],
            })
            for index, task in enumerate(selected, start=4):
                longest = 1
                for col, (field, _, width) in enumerate(columns):
                    value = task[field]
                    if field == '模块':
                        value = value.split(' ', 1)[-1]
                    elif field == '依赖任务':
                        value = value.replace(';', '\n')
                    longest = max(longest, text_lines(value, width))
                    if field == '编号':
                        target = '资料与分工' if name == '未完成任务' else '证据与备注' if name != '证据与备注' else '验收标准'
                        sheet.write_url(index, col, f"internal:'{target}'!A{locations[target][task['编号']]}", fmt['link'], value)
                    elif field in ('最近更新', '完成日期'):
                        if value:
                            sheet.write_datetime(index, col, date.fromisoformat(value), fmt['date'])
                        else:
                            sheet.write_blank(index, col, None, fmt['date'])
                    elif field == '状态':
                        sheet.write_string(index, col, value, fmt['status'][value])
                    elif field == '优先级':
                        sheet.write_string(index, col, value, fmt['priority'][value])
                    else:
                        sheet.write_string(index, col, value, fmt['body'])
                sheet.set_row(index, max(46, longest * 18 + 14))
            sheet.freeze_panes(4, 2)
            sheet.repeat_rows(0, 3)
            sheet.print_area(0, 0, len(selected)+3, last_col)
            sheet.set_selection('C5')
    try:
        temporary.replace(destination)
    except PermissionError as exc:
        raise PermissionError(f'请关闭 Excel 中的 {destination.name} 后重新生成；待保存版本位于 {temporary}') from exc


def check_book(rows: list[dict[str, str]], path: Path = BOOK) -> None:
    """Verify data, source fingerprint and visible layouts after every update."""
    from openpyxl import load_workbook
    if not path.exists():
        raise ValueError('Excel 阅读版不存在，请执行 render')
    book = load_workbook(path, data_only=False)
    try:
        properties = {item.name: item.value for item in book.custom_doc_props}
        if properties.get('TaskTrackerSourceSHA256') != fingerprint(rows):
            raise ValueError('Excel 阅读版与任务数据不一致，请重新生成')
        selections = {'all': rows, 'pending': [r for r in rows if r['状态'] not in CLOSED], 'closed': [r for r in rows if r['状态'] in CLOSED]}
        if book.sheetnames != ['总览与使用说明'] + [spec[0] for spec in SHEETS]:
            raise ValueError('工作表结构不符合阅读版布局')
        for name, scope, columns, _ in SHEETS:
            sheet = book[name]
            selected = selections[scope]
            assert sheet.freeze_panes == 'C5'
            assert sheet.max_row == len(selected)+4
            assert [sheet.cell(4, col+1).value for col in range(len(columns))] == [label for _, label, _ in columns]
            assert len(sheet.tables) == 1 and sheet.sheet_view.showGridLines is False
            for index, task in enumerate(selected, start=5):
                expected_lines = 1
                for col, (field, _, width) in enumerate(columns, start=1):
                    expected = task[field]
                    if field == '模块':
                        expected = expected.split(' ', 1)[-1]
                    elif field == '依赖任务':
                        expected = expected.replace(';', '\n')
                    value = sheet.cell(index, col).value
                    if field in ('最近更新', '完成日期'):
                        actual = value.date().isoformat() if value is not None else ''
                        assert actual == expected, (name, task['编号'], field)
                    else:
                        assert value == expected, (name, task['编号'], field)
                        assert sheet.cell(index, col).alignment.wrap_text, (name, task['编号'], field, 'wrap_text')
                    expected_lines = max(expected_lines, text_lines(expected, width))
                assert sheet.row_dimensions[index].height >= max(46, expected_lines * 18 + 14)
    finally:
        book.close()
