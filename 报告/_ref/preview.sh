#!/usr/bin/env bash
# 用 Word 把 docx/pptx 导出 PDF，再按页统计文字量、出缩略图
set -e
f="$1"; out="$2"
SRC=$(cygpath -w "$(realpath "$f")"); DST=$(cygpath -w "$(realpath -m "$out")")
case "$f" in
  *.docx) powershell.exe -NoProfile -Command "\$w=New-Object -ComObject Word.Application; \$d=\$w.Documents.Open('$SRC'); \$d.SaveAs2('$DST',17); \$d.Close(); \$w.Quit()" ;;
  *.pptx) powershell.exe -NoProfile -Command "\$p=New-Object -ComObject PowerPoint.Application; \$d=\$p.Presentations.Open('$SRC',\$true,\$false,\$false); \$d.SaveAs('$DST',32); \$d.Close(); \$p.Quit()" ;;
esac
python - "$out" <<'PY'
import sys, fitz
d = fitz.open(sys.argv[1])
print("pages", d.page_count, [len(d[i].get_text().strip()) for i in range(d.page_count)])
PY
