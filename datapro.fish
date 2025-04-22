#!/usr/bin/env fish

# 入力CSVファイルとoutput_dirのパスを設定
set csv_file "APTOS_train-val_annotation.csv"
set output_dir "/Volumes/Extreme SSD/aptos2025/APTOS_train-val/chunks"
ß
# output_dirが存在しない場合は作成
if not test -d $output_dir
    mkdir -p $output_dir
end

# CSVファイルを1行ずつ処理（ヘッダーをスキップ）
tail -n +2 $csv_file | while read -l line
    # CSVの各フィールドを取得
    set fields (string split "," $line)
    set video_id $fields[1]
    set start_time $fields[2]
    set end_time $fields[3]

    # video_idからMP4ファイルのパスを構築
    set video_path "/Volumes/Extreme SSD/aptos2025/APTOS_train-val/aptos_videos/$video_id.mp4"
    
    # 出力ファイル名を構築
    set output_file "$output_dir/$video_id"_"$start_time"_"$end_time.mp4"

    # ファイルが存在するか確認
    if test -f $video_path
        echo "Processing: $video_path from $start_time to $end_time"
        echo "========================================="
        echo "Target video: $video_path"
        echo "Time range: $start_time to $end_time"
        echo "Output file: $output_file"
        echo "Command to execute:"
        echo "ffmpeg -i $video_path -ss $start_time -to $end_time -c copy $output_file"
        echo "========================================="
        
        # ffmpegでビデオを切り出し
        ffmpeg -i "$video_path" -ss $start_time -to $end_time -c copy "$output_file"
        
        # エラーチェック
        if test $status -eq 0
            echo "Successfully processed: $output_file"
        else
            echo "Error processing: $video_path"
            echo "Start time: $start_time, End time: $end_time"
            continue
        end
    else
        echo "Video file not found: $video_path"
        continue
    end
end

echo "Processing completed!"