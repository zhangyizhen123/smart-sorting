"""Exercise the real offline audit page with Edge/Playwright, including zoom."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]


def main():
    directory=ROOT/'runs/segment/contour_audit_v1/comparison'
    audit=ROOT/'runs/task_audit';audit.mkdir(exist_ok=True)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    dense=next(c for c in manifest['cases'] if c['name']=='IMG_20261102_002244..jpg')
    errors=[];failed=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        try:
            page=browser.new_page(viewport={'width':1500,'height':1050})
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.on('requestfailed',lambda request:failed.append(request.url))
            page.goto((directory/'index.html').as_uri())
            page.wait_for_function('window.currentCaseKey != null',timeout=20000)
            assert page.locator('#case option').count()==46
            assert not page.locator('#show-box').is_checked()
            assert page.locator('#mask-mode').input_value()=='raw'
            assert '不代表新照片准确率' in page.locator('body').inner_text()
            page.select_option('#case',dense['key'])
            page.wait_for_function('(key)=>window.currentCaseKey===key',arg=dense['key'])
            page.wait_for_function('Array.from(document.querySelectorAll("#panels img")).every(i=>i.complete&&i.naturalWidth>0)')
            assert page.locator('.panel').count()==4
            raw=page.locator('.panel').nth(3).locator('path[data-contour]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("d"))')
            assert raw and page.locator('[data-box]').count()==0
            page.check('#show-box')
            assert page.locator('[data-box]').count()>0
            page.uncheck('#show-box')
            page.select_option('#mask-mode','regularized')
            regular=page.locator('.panel').nth(3).locator('path[data-contour]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("d"))')
            assert len(raw)==len(regular)
            assert raw!=regular, 'A confirmed example must exercise real geometry changes'
            page.select_option('#mask-mode','raw')
            restored=page.locator('.panel').nth(3).locator('path[data-contour]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("d"))')
            assert restored==raw, 'Switching back must restore the exact original paths'
            page.select_option('#size','960')
            assert '输入960' in page.locator('#status').inner_text()
            page.select_option('#size','640')
            page.screenshot(path=str(audit/'contour_audit_desktop.png'))
            page.locator('#panels').screenshot(path=str(audit/'contour_audit_dense.png'))
            page.locator('.panel').nth(3).locator('.objects button').first.click()
            assert page.locator('#zoom').is_visible()
            assert float(page.locator('#zoom-level').input_value())>=2
            assert page.locator('#zoom-stage img').count()==1
            page.screenshot(path=str(audit/'contour_audit_object_zoom_raw.png'))
            page.click('#zoom-close');assert not page.locator('#zoom').is_visible()
            page.select_option('#mask-mode','regularized')
            page.locator('.panel').nth(3).locator('.objects button').first.click()
            page.screenshot(path=str(audit/'contour_audit_object_zoom_regularized.png'))
            page.click('#zoom-close')
            page.select_option('#mask-mode','raw')
            for group,count in [('empty_tray',11),('white_cube',12),('multi_object',15),('occlusion',1)]:
                page.select_option('#group',group)
                page.wait_for_function('window.currentCaseKey != null')
                assert page.locator('#case option').count()==count
            for domain,count in [('legacy',108),('focused',90),('all',244)]:
                page.select_option('#domain',domain)
                page.wait_for_function('window.currentCaseKey != null')
                assert page.locator('#case option').count()==count
            page.select_option('#domain','real');page.select_option('#group','occlusion')
            page.wait_for_function('(key)=>window.currentCaseKey===key',arg=dense['key'])
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.screenshot(path=str(audit/'contour_audit_mobile.png'))
            # Verify every local image and sidecar exists, without eagerly decoding 244 full photos.
            for case in manifest['cases']:
                assert (directory/'data'/(case['key']+'.js')).is_file()
                assert (directory/'images'/(case['key']+'.jpg')).is_file()
            assert not errors,errors
            assert not failed,failed
        finally:
            browser.close()
    result={'cases':244,'variants':4,'default_original_contours':True,'original_roundtrip_exact':True,
            'box_toggle':True,'regularized_toggle':True,'object_zoom':True,'groups':[11,12,15,1],
            'mobile_horizontal_overflow':False,'javascript_errors':errors,'failed_requests':failed}
    (audit/'2026-11-01_contour_audit_ui.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
