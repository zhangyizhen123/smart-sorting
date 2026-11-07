"""Check that all fixed pilot comparison images load without layout clipping."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]


def main():
    audit=ROOT/'runs/task_audit';errors=[]
    path=ROOT/'runs/segment/real_mixed_pilot_v1/comparison/index.html'
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(path.as_uri());page.locator('section').first.wait_for()
        assert page.locator('section').count()==8 and page.locator('img').count()==32
        page.locator('img').evaluate_all('(images)=>images.forEach(i=>i.loading="eager")')
        page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)',timeout=15000)
        assert '不代表新照片准确率' in page.locator('body').inner_text()
        page.screenshot(path=str(audit/'real_pilot_comparison_desktop.png'))
        page.locator('section').nth(6).screenshot(path=str(audit/'real_pilot_dense_scene.png'))
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.screenshot(path=str(audit/'real_pilot_comparison_mobile.png'))
        assert not errors,errors
        browser.close()
    result={'cases':8,'images_loaded':32,'console_errors':errors,'mobile_horizontal_overflow':False,'training_fit_notice_visible':True}
    (audit/'2026-11-01_real_pilot_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
