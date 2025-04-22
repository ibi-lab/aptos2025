#!/usr/bin/env fish

set CMD "python ./datam.py $argv"
while true
    echo "実行: $CMD"
    eval $CMD
    set ret $status
    if test $ret -eq 0
        echo "成功しました。"
        break
    else
        echo "datam.py がエラー終了しました（終了コード $ret）。再実行します..."
        sleep 5
    end
end