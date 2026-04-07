import pandas as pd
import glob
import os

def analyze_all_data_kpis(folder_path="."):
    # 1. Tìm tất cả các file *_in_ngabavoi.csv hoặc tương tự
    all_files = glob.glob(os.path.join(folder_path, "*_debug_ngabavoi.csv"))
    
    if not all_files:
        print(f"[-] Không tìm thấy file dữ liệu nào tại: {os.path.abspath(folder_path)}")
        return

    summary_results = []

    for filename in all_files:
        try:
            # Đọc CSV với header mới: Time, Lane, Delay_s, Saturation, Avg_Queue_Veh, Inflow_PCU_h, TotalDemand_h, Capacity_h, Residual_NGE
            df = pd.read_csv(filename)
            if df.empty: continue

            df_to_analyze = df.copy()

            # Xác định kịch bản từ tên file
            case_name = os.path.basename(filename).replace(".csv", "").upper()
            
            # Tính toán các chỉ số trung bình dựa trên cơ chế tồn dư mới
            metrics = {
                "Scenario": case_name,
                "Avg_Delay_s": df_to_analyze['Delay_s'].mean(),
                "Avg_Saturation": df_to_analyze['Saturation'].mean(),
                "Avg_Queue_Veh": df_to_analyze['Avg_Queue_Veh'].mean(),
                "Avg_Inflow_PCU": df_to_analyze['Inflow_PCU_h'].mean(),
                "Avg_TotalDemand": df_to_analyze['TotalDemand_h'].mean(),
                "Avg_Residual_NGE": df_to_analyze['Residual_NGE'].mean(),
                "Avg_Capacity": df_to_analyze['Capacity_h'].mean(),
                "Total_Cycles": len(df_to_analyze) // df_to_analyze['Lane'].nunique() # Ước tính số chu kỳ
            }
            summary_results.append(metrics)
            
        except Exception as e:
            print(f"[!] Lỗi khi xử lý file {os.path.basename(filename)}: {e}")

    if not summary_results:
        print("[-] Không có kết quả nào được tổng hợp.")
        return

    # 2. Tạo DataFrame so sánh
    summary_df = pd.DataFrame(summary_results)
    
    # Sắp xếp theo Delay trung bình
    summary_df = summary_df.sort_values(by="Avg_Delay_s")

    print("\n" + "="*125)
    print(f"{'KỊCH BẢN':<18} | {'DELAY TB':<10} | {'V/C (SAT)':<10} | {'Q_AVG (xe)':<10} | {'INFLOW':<10} | {'DEMAND':<10} | {'RESIDUAL'}")
    print("-" * 125)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Scenario']:<18} | {row['Avg_Delay_s']:>8.2f}s | {row['Avg_Saturation']:>10.2f} | {row['Avg_Queue_Veh']:>10.2f} | {row['Avg_Inflow_PCU']:>10.2f} | {row['Avg_TotalDemand']:>10.2f} | {row['Avg_Residual_NGE']:>10.2f}")
    
    print("-" * 125)
    print(f"[*] Báo cáo tổng hợp từ {len(summary_df)} tệp dữ liệu.")
    print("="*125 + "\n")

    # Xuất file báo cáo tổng hợp mới
    summary_df.to_csv("summary_analyzer_ngabavoi.csv", index=False)
    print("[*] Đã lưu báo cáo phân tích tại: summary_analyzer_ngabavoi.csv")

if __name__ == "__main__":
    analyze_all_data_kpis()
