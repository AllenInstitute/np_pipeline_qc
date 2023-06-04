from __future__ import annotations

import html
import itertools
import logging
import pathlib
import re
import shutil


import importlib_resources as resources
import bs4
import np_session

from np_pipeline_qc.reports.assets.html_templates import DOC, FIG_TAG, JSON_TAG

CORBETT_QC_ROOT = pathlib.Path(
    '//allen/programs/braintv/workgroups/nc-ophys/corbettb/NP_behavior_pipeline/QC'
)
OPENSCOPE_ROOT = pathlib.Path(
    '//allen/programs/mindscope/workgroups/openscope'
)

QC_ROOTS = (CORBETT_QC_ROOT, *tuple(OPENSCOPE_ROOT.glob('*_QC')))

NPEXP = pathlib.Path('//allen/programs/mindscope/workgroups/np-exp')

HTML_FILENAME = 'qc'
SHORTCUT_FILENAME = 'QC'
CSS_FILENAME = 'single_page_img_json_report'

CSS = (
    resources.files(__package__ or 'np_pipeline_qc')
    / 'assets'
    / f'{CSS_FILENAME}.css'
)
# hard-coded package name will be used if module is run as __main__


def session_qc_dir_to_img_html(session_qc_root: str | pathlib.Path) -> None:
    session = np_session.Session(str(session_qc_root))
    session_qc_root = session.qc_path
    doc = bs4.BeautifulSoup(DOC, features='html.parser')
    doc.head.title.append(f'{session} {session_qc_root.parent.name}')
    doc.head.h1.append(f'{session} {session_qc_root.parent.name}')
    doc.head.link['href'] = f'{CSS_FILENAME}.css'

    def fmt(string: str) -> str:
        new = ''.join(string)
        for s in session.folder.split('_'):
            new = new.replace(s, '_')
        return new.replace('_', ' ').strip()

    def add_section(p: pathlib.Path, heading_idx: int = 1) -> bs4.Tag:
        if p.parent == session_qc_root:
            parent = doc.head
        else:
            parent = doc.body
        parent.append(title := doc.new_tag(f'h{heading_idx}'))
        title.append(fmt(p.name) if p.is_dir() else fmt(p.parent.name))
        parent.append(section := doc.new_tag(f'div', attrs={'class': 'row'}))
        return section

    def add_figure(p: pathlib.Path, section: bs4.Tag) -> None:
        section.append(div := doc.new_tag(f'div', attrs={'class': 'column'}))
        fig = bs4.BeautifulSoup(FIG_TAG, features='html.parser')
        fig.img['src'] = fig.img['alt'] = p
        fig.figcaption.b.a.append(fmt(p.stem))
        fig.figcaption.b.a['href'] = Rf'file:///{p}'
        div.append(fig)

    def add_json(p: pathlib.Path, section: bs4.Tag) -> None:
        if 'plotly' in p.name:
            return
        div = doc.new_tag(f'div', attrs={'class': 'column'})
        section.append(div)
        json = bs4.BeautifulSoup(JSON_TAG, features='html.parser')
        json.iframe['src'] = json.iframe['title'] = p
        json.figcaption.b.a.append(fmt(p.stem))
        json.figcaption.b.a['href'] = Rf'file:///{p}'
        div.append(json)

    def add_qc_contents(path: pathlib.Path, idx: int = 1):
        """Recursively add subfolders and their files to html doc"""
        section = None
        for p in path.iterdir():
            if '.vscode' in p.parts:
                continue
            if p.is_dir():
                add_qc_contents(p, idx + 1)
            else:
                if not section:
                    section = add_section(p, idx)
                if p.suffix == '.png':
                    add_figure(p, section)
                elif p.suffix == '.json':
                    add_json(p, section)

    add_qc_contents(session_qc_root)
    save_path: pathlib.Path = session_qc_root / f'{HTML_FILENAME}.html'
    save_path.touch()
    # print(doc.prettify())
    save_path.write_bytes(doc.encode())
    shutil.copy(CSS, session_qc_root / CSS.name)
    print(f'Saved {save_path.relative_to(save_path.parent.parent.parent)}')



def get_session_folder(path: str | pathlib.Path) -> str | None:
    """Extract [8+digit session ID]_[6-digit mouse ID]_[8-digit date
    str] from a string or path"""
    session_reg_exp = R'[0-9]{8,}_[0-9]{6}_[0-9]{8}'

    session_folders = re.findall(session_reg_exp, str(path))
    if session_folders:
        if not all(s == session_folders[0] for s in session_folders):
            logging.debug(
                f'Mismatch between session folder strings - file may be in the wrong folder: {path}'
            )
        return session_folders[0]
    return None


def make_qc_html_for_most_recent_sessions():
    all_qc_paths = tuple(
        itertools.chain.from_iterable(r.iterdir() for r in QC_ROOTS)
    )
    for npexp_session_path in sorted(
        list(filter(get_session_folder, NPEXP.iterdir())), reverse=True
    ):
        if npexp_session_path.name in (
            qc_path.name for qc_path in all_qc_paths
        ):
            if not tuple(npexp_session_path.glob(f'{SHORTCUT_FILENAME}*.lnk')):
                matching_qc_paths = (
                    qc_path
                    for qc_path in all_qc_paths
                    if qc_path.name == npexp_session_path.name
                )
                for qc_path in matching_qc_paths:
                    session_qc_dir_to_img_html(qc_path)
            else:
                # upon reaching the first session folder on npexp that has QC and
                # a shortcut to qc.html
                return


def make_qc_html_for_all_existing_qc():
    for d in itertools.chain.from_iterable(r.iterdir() for r in QC_ROOTS):
        if (
            d.is_dir()
            and (session := get_session_folder(d))
            and (NPEXP / session).exists()
        ):
            session_qc_dir_to_img_html(d)


if __name__ == '__main__':
    make_qc_html_for_most_recent_sessions()
