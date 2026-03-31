Các file cần thiết để tính kpi, lưu trữ KPI của từng kịch bản, so sánh
- Cần tạo lại các file rou phù hợp thực tế làm dữ liệu đầu vào, lấy từ video đầu vào
- kpi_engine chứa các công thức tính kpi
- run_kpi: để nhận dữ liệu tủ, đẩy dữ liệu detector, thực hiện tính các kpi
    - Chỉnh lại tên file lưu trữ các kpi
    - chỉnh lại dữ liệu id tủ, mess gửi tủ, tên file mapping import vào
- kpi_engine: so sánh các dữ liệu của các file .cvs -> plot ra so sánh


Đang dùng new_main.py để test việc mapping, tính kpi theo engine chuẩn -> mai thực hiện test, cần các file liên kết 
- config_master.json chứa thông tin tủ, thông tin config tính kpi (quan trọng có cycle_len, kpi_ pcu mapping, detector)
- file lamp_mapping: dữ nguyên để mapping tủ với đèn
- file engine chứa các công thức
-> file test hiện đang thử nghiệm:
- đánh giá đã đồng bộ đèn tủ chưa
- đã lên tín hiệu detector chưa
- đánh giá kpi tính đúng không
-> áp dụng chuẩn hóa với các ngã tư khác