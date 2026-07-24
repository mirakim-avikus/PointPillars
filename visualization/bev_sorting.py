import os
import re
import shutil

def process_metrics(root_path):
    output_dir = "BEV_sorting"
    video_root_dir = os.path.join(output_dir, "BEV_sorting_video")
    
    # 01, 02, 03 스코어 폴더 정의 (영상 복사 대상)
    SCORE_TAGS = ["01", "02", "03"]
    
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # 1. 데이터 수집 (sorting 기준인 metric.txt는 공통이므로 기존 로직 유지)
    for subdir in os.listdir(root_path):
        subdir_path = os.path.join(root_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        metric_file = os.path.join(subdir_path, "metric", "metric.txt")
        
        if os.path.exists(metric_file):
            with open(metric_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # [metric values_m] 섹션에서 mAPBEV 값 추출
                match = re.search(r'\[metric values_m\].*?mAPBEV=([\d.]+)', content, re.DOTALL)
                
                if match:
                    bev_val = float(match.group(1))
                    results.append({
                        'name': subdir,
                        'val': bev_val,
                        'path': subdir_path
                    })

    # 2. 내림차순 정렬
    results.sort(key=lambda x: x['val'], reverse=True)

    # 3. BEV_sorting.txt 저장
    txt_path = os.path.join(output_dir, "BEV_sorting.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, res in enumerate(results, 1):
            rounded_val = round(res['val'], 2)
            f.write(f"{i}. {res['name']} : {rounded_val}\n")
            
            # 4. 점수대별(01, 02, 03) 영상 파일 복사
            for tag in SCORE_TAGS:
                # 원본 경로: subdir/bev/01/gt_pred_noimu.mp4
                video_src = os.path.join(res['path'], "bev", tag, "gt_pred_noimu.mp4")
                
                if os.path.exists(video_src):
                    # 목적지 경로: BEV_sorting_video/01/
                    dst_subdir = os.path.join(video_root_dir, tag)
                    os.makedirs(dst_subdir, exist_ok=True)
                    
                    new_video_name = f"{i}-{res['name']}-{rounded_val}.mp4"
                    video_dst = os.path.join(dst_subdir, new_video_name)
                    
                    shutil.copy2(video_src, video_dst)

    print(f"--- 작업 완료 ---")
    print(f"텍스트 결과: {txt_path}")
    print(f"영상 분류 완료: {video_root_dir} (하위 01, 02, 03 폴더 확인)")

if __name__ == "__main__":
    base_path = './data_to_superbai_1114'
    process_metrics(base_path)