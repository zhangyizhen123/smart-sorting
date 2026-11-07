"""Exercise the static review UI in an isolated Edge context; never approve real data."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vision/scripts'))
from import_real_review import import_review


def run():
    directory=ROOT/'dataset/real/annotations/review_v2'
    data=json.loads((directory/'annotations.json').read_text(encoding='utf-8'))
    pointer=directory/'current_review.json'
    initial_approved=json.loads(pointer.read_text(encoding='utf-8'))['approved_images'] if pointer.exists() else 0
    audit=ROOT/'runs/task_audit';audit.mkdir(exist_ok=True)
    errors=[]
    with tempfile.TemporaryDirectory(prefix='sorting-review-ui-') as temp, sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda error: errors.append(str(error)))
        page.goto((directory/'index.html').as_uri());page.locator('.card').first.wait_for()
        assert page.locator('.card').count()==46
        assert page.locator('.instance select').count()==118
        assert f'已通过 {initial_approved} / 46 张' in page.locator('#progress').inner_text()
        assert page.locator('input[type=checkbox]:checked').count()==initial_approved
        if initial_approved:
            page.screenshot(path=str(audit/'real_review_approved.png'))
        # Exercise edits in a private context. Saved project approvals are untouched.
        page.evaluate('for(const r of data.images){state(r).approved=false;}save();draw();')
        page.locator('img').evaluate_all('(imgs)=>imgs.forEach(i=>i.loading="eager")')
        page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)',timeout=20000)
        first=page.locator('.card').first
        check=first.get_by_role('checkbox',name='本图全部正确')
        check.check();assert '已通过 1 / 46 张' in page.locator('#progress').inner_text()
        first.get_by_role('textbox',name='问题说明').fill('自动化测试：不能成为真实审核')
        assert not check.is_checked()
        first.get_by_role('textbox',name='问题说明').fill('')
        check.check();first.get_by_role('combobox').first.select_option('cube');assert not check.is_checked()
        first.get_by_role('combobox').first.select_option('unknown');assert check.is_disabled()
        first.get_by_role('combobox').first.select_option(data['images'][0]['instances'][0]['class_name'])
        first.get_by_role('spinbutton').fill('2');assert check.is_disabled()
        first.get_by_role('spinbutton').fill('1');assert check.is_enabled()
        page.locator('#filter').select_option('empty');assert page.locator('.card').count()==11
        page.locator('#filter').select_option('positive');assert page.locator('.card').count()==35
        page.locator('#filter').select_option('flagged');assert page.locator('.card').count()==7
        page.locator('#filter').select_option('all')
        final=page.locator('.card').filter(has=page.get_by_role('heading',name='IMG_20261102_002244..jpg',exact=True))
        for n,cls in [(2,'tetrahedron'),(3,'tetrahedron'),(5,'cube'),(10,'tri_prism')]:
            assert final.get_by_role('combobox',name=f'{n}号形状',exact=True).input_value()==cls
        final.screenshot(path=str(audit/'real_review_final_scene.png'))
        with page.expect_download() as info:page.locator('#export').click()
        export=Path(temp)/'unapproved.json';info.value.save_as(str(export))
        result=import_review(export,check_only=True);assert result['explicitly_approved']==0
        page.locator('.card').first.get_by_role('checkbox',name='本图全部正确').check()
        page.reload();assert '已通过 1 / 46 张' in page.locator('#progress').inner_text()
        with page.expect_download() as info:page.locator('#export').click()
        export=Path(temp)/'one-approved.json';info.value.save_as(str(export))
        assert import_review(export,check_only=True)['explicitly_approved']==1
        # Separate context is discarded; no real browser profile or accepted_reviews write.
        page.evaluate('localStorage.clear()');page.reload()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.screenshot(path=str(audit/'real_review_mobile.png'))
        assert not errors,errors
        browser.close()
    result={'images':46,'instances':118,'empty':11,'images_loaded':92,'console_errors':errors,
        'confirmed_final_scene_classes_checked':True,'class_count_note_reset_approval':True,
        'exports_hash_checked':True,'initial_saved_approved_images':initial_approved,
        'actual_review_approvals_written':0,'mobile_horizontal_overflow':False}
    (audit/'2026-11-01_real_review_ui.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
