#!/bin/bash
# ============================================
# 一键提交并推送到 GitHub
#
# 用法：
#   bash push.sh "提交信息"
# 如果不填提交信息，会自动用「更新 + 日期时间」作为默认信息
# ============================================

# 提交信息：取第一个参数，为空则用默认
MSG="${1:-更新 $(date '+%Y-%m-%d %H:%M')}"

echo ">>> 当前改动："
git status -s
echo ""

echo ">>> 提交信息：$MSG"
git add .
git commit -m "$MSG"
echo ""

echo ">>> 推送到 GitHub..."
git push
echo ""

echo ">>> 完成！"
