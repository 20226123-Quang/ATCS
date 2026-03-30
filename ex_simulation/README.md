Các folder đều chứa các file:
- cal_crc.py để tính checksum dữ liệu gửi đến tủ
- debug.py để nghe mã phase tương ứng với dữ liệu đèn nào (chỉnh theo tủ tương ứng)
- detector_config.json
    - chỉnh mapping tương ứng giữa detector id và phase được call 
    - chỉnh id tủ tương ứng
- *_mapping.py chứa mapping phase đèn (nghe được ở debug) và tín hiệu đèn tương ứng truyền vào sumo
    - nghe và chỉnh mapping lại
    - hiện đnag bỏ qua stage vàng
    - đổi tls_id tương ứng (tra bằng keyword tlLogic id)
- main_simulation.py
    - chứa dữ liệu config
    - chỉnh phần import *_mapping sang tên tương ứng
    - chỉnh id tủ, cross tương ứng
    - lệnh khởi động sumo -> đổi tên .sumocfg tương ứng


--công việc còn
- 2 nút giao config lấy dữ liệu 2 tủ
- file đánh giá kpi -> áp dụng cho từng nút giao, từng kịch bản
- dữ liệu 
- check file .rou datareal
-> lưu kpi đánh giá -> bảng biểu đánh giá