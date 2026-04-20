import pandas as pd
import glob
import os

def summarize_case_performance(folder_path="."):
    # 1. Tìm tất cả các file kết quả ngã ba voi
    all_files = glob.glob(os.path.join(folder_path, "crowd_*_ngabavoi.csv"))
    
    if not all_files:
        print("[-] Không tìm thấy file dữ liệu nào.")
        return

    overall_results = []

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Làm sạch tên cột (tránh lỗi khoảng trắng/BOM)
            df.columns = [c.strip() for c in df.columns]
            
            case_name = os.path.basename(filename).replace("crowd_", "").replace("_ngabavoi.csv", "").upper()
            
            # Chỉ tính toán trên các dòng có lưu lượng thực tế
            df_active = df[df['Inflow_PCU_h'] > 0].copy()
            
            if df_active.empty:
                continue

            # --- TÍNH TOÁN THEO TRỌNG SỐ LƯU LƯỢNG (Chuẩn TCCS 24) ---
            total_inflow = df_active['Inflow_PCU_h'].sum()
            
            # 1. Trễ trung bình nút giao (s/xe) = sum(delay * inflow) / sum(inflow)
            weighted_delay = (df_active['Delay_s'] * df_active['Inflow_PCU_h']).sum() / total_inflow
            
            # 2. Độ bão hòa trung bình (v/c)
            avg_sat = df_active['Saturation'].mean()
            
            # 3. Hàng chờ vật lý trung bình (m) - Tính từ Avg_Queue thực tế
            # Vì bạn dùng unique_lanes, tổng Avg_Queue nhân 7.5 sẽ ra chiều dài hàng đợi thực tế tại nút
            avg_junction_queue = df_active['Avg_Queue'].mean() * 7.5
            
            # 4. Tổng năng lực thông hành (Throughput)
            total_outflow = df_active['Outflow_PCU_h'].mean()

            overall_results.append({
                "Kịch bản": case_name,
                "Trễ TB Nút (s/xe)": round(weighted_delay, 2),
                "Độ bão hòa TB": round(avg_sat, 2),
                "Hàng chờ TB (m)": round(avg_junction_queue, 2),
                "Lưu lượng TB (PCU/h)": round(total_outflow, 2),
                "Số mẫu": len(df_active)
            })
            
        except Exception as e:
            print(f"[!] Lỗi xử lý {filename}: {e}")

    # 2. Xuất bảng so sánh
    summary_df = pd.DataFrame(overall_results).sort_values(by="Trễ TB Nút (s/xe)")
    
    print("\n" + "="*95)
    print(f"{'CHẾ ĐỘ':<15} | {'TRỄ TB (s/xe)':<15} | {'V/C AVG':<12} | {'Q_LENGTH (m)':<15} | {'THROUGHPUT'}")
    print("-" * 95)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Kịch bản']:<15} | {row['Trễ TB Nút (s/xe)']:>13.2f}s | {row['Độ bão hòa TB']:>11.2f} | {row['Hàng chờ TB (m)']:>13.2f}m | {row['Lưu lượng TB (PCU/h)']:>10.2f}")
    
    print("="*95)
    
    # 3. Đánh giá LOS (Level of Service) theo TCCS 24:2018
    print("\n[ĐÁNH GIÁ MỨC PHỤC VỤ (LOS) DỰA TRÊN TRỄ TB]:")
    for _, row in summary_df.iterrows():
        d = row['Trễ TB Nút (s/xe)']
        los = "A" if d <= 10 else "B" if d <= 20 else "C" if d <= 35 else "D" if d <= 55 else "E" if d <= 80 else "F"
        print(f" -> Kịch bản {row['Kịch bản']}: LOS {los} ({'Tốt' if los in 'ABC' else 'Cần cải thiện'})")

if __name__ == "__main__":
    summarize_case_performance()