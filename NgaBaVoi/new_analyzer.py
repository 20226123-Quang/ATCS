import pandas as pd
import glob
import os

def analyze_all_data_kpis(folder_path="."):
    # 1. Tìm tất cả các file crowd_*_ngabavoi.csv
    all_files = glob.glob(os.path.join(folder_path, "*_*_ngabavoi.csv"))
    
    if not all_files:
        print(f"[-] Không tìm thấy file dữ liệu nào tại: {os.path.abspath(folder_path)}")
        return

    summary_results = []

    for filename in all_files:
        try:
            # Đọc CSV với đúng header 8 cột của bạn
            df = pd.read_csv(filename)
            if df.empty: continue

            # KHÔNG DÙNG BỘ LỌC - Lấy toàn bộ dữ liệu để tính toán
            df_to_analyze = df.copy()

            # Xác định kịch bản từ tên file
            case_name = os.path.basename(filename).replace(".csv", "").upper()
            
            # Tính toán các chỉ số trung bình trên toàn bộ tập dữ liệu
            metrics = {
                "Scenario": case_name,
                "Avg_Delay_s": df_to_analyze['Delay_s'].mean(),
                "Avg_Saturation": df_to_analyze['Saturation'].mean(),
                "Avg_Queue_Veh": df_to_analyze['Avg_Queue'].mean(),
                "Avg_Queue_m": df_to_analyze['Queue_m'].mean(),
                "Avg_Inflow_PCU": df_to_analyze['Inflow_PCU_h'].mean(),
                "Avg_Outflow_PCU": df_to_analyze['Outflow_PCU_h'].mean(),
                "Total_Records": len(df_to_analyze)
            }
            summary_results.append(metrics)
            
        except Exception as e:
            print(f"[!] Lỗi khi xử lý file {os.path.basename(filename)}: {e}")

    if not summary_results:
        print("[-] Không có kết quả nào được tổng hợp.")
        return

    # 2. Tạo DataFrame so sánh
    summary_df = pd.DataFrame(summary_results)
    
    # Sắp xếp theo Delay trung bình (thấp nhất là tối ưu nhất)
    summary_df = summary_df.sort_values(by="Avg_Delay_s")

    print("\n" + "="*110)
    print(f"{'KỊCH BẢN':<15} | {'DELAY TB':<12} | {'V/C (SAT)':<12} | {'Q_AVG (xe)':<12} | {'INFLOW TB':<12} | {'OUTFLOW TB'}")
    print("-" * 110)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Scenario']:<15} | {row['Avg_Delay_s']:>10.2f}s | {row['Avg_Saturation']:>12.2f} | {row['Avg_Queue_Veh']:>12.2f} | {row['Avg_Inflow_PCU']:>12.2f} | {row['Avg_Outflow_PCU']:>12.2f}")
    
    print("-" * 110)
    print(f"[*] Báo cáo so sánh dựa trên tổng số {len(summary_df)} kịch bản.")
    print("="*110 + "\n")

    # Xuất file báo cáo tổng hợp
    summary_df.to_csv("all_records_ngabavoi.csv", index=False)
    print("[*] Đã lưu báo cáo đầy đủ tại: all_records_ngabavoi.csv")

if __name__ == "__main__":
    analyze_all_data_kpis()