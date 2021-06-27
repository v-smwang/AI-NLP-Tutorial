import os
import re


def get_chapter_start_html(chapter):
    chapter_start_html = """
    <?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-CN">
    <head>
    <title>"""+ chapter +"""</title>
    <link href="stylesheet.css" type="text/css" rel="stylesheet"/><style type="text/css">
    @page { margin-bottom: 5.000000pt; margin-top: 5.000000pt; }</style>
    </head>
    <body>
    """
    return chapter_start_html

chapter_end_html = """<div class="mbppagebreak"></div></body></html>"""

def get_chapter_html(chapter):
    chapter_html = """<h2 align="center"><span style="border-bottom:1px solid">""" + chapter + """</span></h2>"""
    return chapter_html

def get_paragraph_html(paragraph):
    paragraph_html = """<p>""" + paragraph + """</p>"""
    return paragraph_html

def get_catalog_start_html(fiction_name):
    catalog_start_html = """
    <?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-CN">
    <head>
    <title>"""+ fiction_name +"""</title>
    <link href="stylesheet.css" type="text/css" rel="stylesheet"/><style type="text/css">
    @page { margin-bottom: 5.000000pt; margin-top: 5.000000pt; }</style>
    </head>
    <body>
    <h1>目录<br/>Content</h1>
    <ul>
    """
    return catalog_start_html

def get_catalog_link_html(catalog):
    catalog_link_html = """<li class="catalog"><a href=\""""+catalog+""".html">"""+ catalog +"""</a></li>"""
    return catalog_link_html

catalog_end_html = """</ul><div class="mbppagebreak"></div></body></html>"""    
f = open('/Users/a1/Documents/liwen/new.txt', 'r+', encoding='utf-8')
catalogs = []
for line in f.readlines():
    if re.match('^第.*章.*',line):
#         print(line)
        if file:
            file.write(chapter_end_html)
        catalogs.append(line.strip())
        file = open('/Users/a1/Documents/liwen/'+line.strip()+'.html', 'a+', encoding='utf-8')
        chapter = line
        file.write(get_chapter_start_html(chapter))
        file.write(get_chapter_html(chapter))
    elif file:
        paragraph = line
        file.write(get_paragraph_html(paragraph))
with open('/Users/a1/Documents/liwen/catalog.html', 'a+', encoding='utf-8') as f:    
    for catalog in catalogs:
        f.write(get_catalog_link_html(catalog))
