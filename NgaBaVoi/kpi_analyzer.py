import pandas as pd
import glob
import os

def analyze_kpis(folder_path="."):
    # 1. Tìm tất cả các file kpi_*.csv trong thư mục hiện tại hoặc results/
    all_files = glob.glob(os.path.join(folder_path, "*_ngabavoi.csv"))
    
    if not all_files:
        print(f"[-] Không tìm thấy file *_ngabavoi.csv nào trong {os.path.abspath(folder_path)}")
        return

    summary_results = []

    for filename in all_files:
        try:
            # Đọc CSV
            df = pd.read_csv(filename)
            if df.empty: continue

            # Tên trường hợp (VD: kpi_fix_time.csv -> fix_time)
            case_name = os.path.basename(filename).replace("kpi_", "").replace(".csv", "")
            
            # Tính toán trung bình toàn cục cho kịch bản này
            # Cần đảm bảo các tên cột này khớp 100% với header trong file CSV của bạn
            metrics = {
                "Case": case_name,
                "Avg_Delay": df.iloc[:, 2].mean(),    # Cột 3: Delay_s
                "Max_Queue": df.iloc[:, 3].mean(),     # Cột 4: Queue_m
                "Avg_Saturation": df.iloc[:, 4].mean(), # Cột 5: Saturation
                "Records": len(df)
            }
            summary_results.append(metrics)
        except Exception as e:
            print(f"[!] Lỗi khi đọc file {filename}: {e}")

    if not summary_results:
        print("[-] Không có dữ liệu hợp lệ để phân tích.")
        return

    # 2. Tạo DataFrame tổng hợp
    summary_df = pd.DataFrame(summary_results)
    
    # 3. Sắp xếp theo Avg_Delay (Thằng nào thấp nhất là tốt nhất)
    summary_df = summary_df.sort_values(by="Avg_Delay")
    
    print("\n" + "="*75)
    print(f"{'CHẾ ĐỘ ĐIỀU KHIỂN':<25} | {'DELAY TB (s)':<12} | {'HÀNG CHỜ MAX':<12} | {'SỐ MẪU'}")
    print("-" * 75)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Case']:<25} | {row['Avg_Delay']:<12.2f} | {row['Max_Queue']:<12.2f} | {row['Records']}")
    
    print("="*75)
    
    # Xuất báo cáo tổng hợp
    summary_df.to_csv("comparison_report.csv", index=False)
    print(f"[+] Đã xuất báo cáo so sánh tại: {os.getcwd()}/comparison_report.csv")

if __name__ == "__main__":
    analyze_kpis()