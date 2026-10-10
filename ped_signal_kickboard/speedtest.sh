#!/usr/bin/env bash
# AI Hub 다운로드 속도: 1개 / 3개 동시 (각 30초)
K=$(cat ~/work/code/aihub_apikey.txt)
U="https://api.aihub.or.kr/down/0.6"
one() { curl -s -L -o /dev/null -m 30 -H "apikey:$K" -w "%{speed_download}\n" "$U/$1"; }
echo "single: $(one '71579.do?fileSn=497022') B/s"
a=$(one '71579.do?fileSn=497022' &) ; :
r1=$(mktemp); r2=$(mktemp); r3=$(mktemp)
one '188.do?fileSn=49280' > $r1 & one '188.do?fileSn=49281' > $r2 & one '614.do?fileSn=56583' > $r3 & wait
echo "parallel3: $(cat $r1) $(cat $r2) $(cat $r3) B/s"
