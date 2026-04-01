#!/bin/bash
# SSSD后台运行脚本

cd /home/rhl/Github

nohup python baselines/pm25_sssd_forecast.py --epochs 50 --device cuda:0 > ./save/sssd_train.log 2>&1 &

echo "SSSD已在后台启动"
echo "日志文件: ./save/sssd_train.log"
echo "查看日志: tail -f ./save/sssd_train.log"
echo "查看进程: ps aux | grep pm25_sssd"
