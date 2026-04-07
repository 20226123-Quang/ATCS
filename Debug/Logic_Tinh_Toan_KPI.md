# Hướng dẫn Logic Tính toán KPI (Lưu lượng & Thời gian xanh)

Tài liệu này giải thích cách hệ thống `KPIEngine` và `main.py` phối hợp để đo lường các thông số đầu vào cho công thức tính toán KPI theo tài liệu chuẩn.

---

## 1. Cách tính Lưu lượng (Flow)

Hệ thống phân chia lưu lượng thành hai loại để đảm bảo độ chính xác cao nhất:

### A. Input Flow (Lưu lượng vào làn)
- **Thời điểm ghi nhận:** Ngay khi một chiếc xe xuất hiện trên làn đường tiếp cận (Approach Lane) của nút giao.
- **Cách đếm:** 
    - Mỗi xe có một `v_id` duy nhất. Hệ thống sử dụng một tập hợp `v_ids_in_cycle` để lưu danh sách các xe đã vào làn trong chu kỳ hiện tại.
    - Nếu xe mới xuất hiện (chưa có trong tập hợp), hệ thống sẽ:
        1. Tra cứu loại xe để lấy hệ số **PCU** (xe máy=0.5, ô tô=1.0...).
        2. Sử dụng hàm `getNextLinks()` để biết xe này định đi theo tín hiệu đèn (tlsid index) nào.
        3. Cộng dồn PCU vào `link_inflow_pcu` của đúng `index` đó.
- **Mục đích:** Để tính toán mức độ bão hòa (Saturation) và dự báo hàng đợi tích lũy.

### B. Outflow / Thoát xe (Lưu lượng ra khỏi làn)
- **Thời điểm ghi nhận:** Khi xe băng qua vạch dừng và rời khỏi làn đường tiếp cận để đi vào nút giao.
- **Cách đếm:** 
    - Hệ thống so sánh danh sách xe ở bước hiện tại và bước trước đó.
    - Những xe có ở bước trước nhưng mất đi ở bước hiện tại được coi là **"Thoát xe"**.
    - PCU của xe thoát sẽ được cộng vào `link_outflow_pcu` của `index` đèn tương ứng.
- **Mục đích:** Đây là giá trị $q_i$ dùng để tính tỷ lệ dòng $a_i$ và lưu lượng bão hòa hỗn hợp $S_{hh}$ theo công thức (F-3).

---

## 2. Cách tính Thời gian xanh (Green Time)

Thời gian xanh được đếm "thực tế" theo từng bước nhảy của mô phỏng (mặc định là 0.5 giây/bước).

### A. Thời gian xanh theo dòng (Link Green Time)
- Trong mỗi bước nhảy, hệ thống kiểm tra chuỗi trạng thái đèn đang áp dụng (ví dụ: `GGGrrr...`).
- Với mỗi làn đường, hệ thống biết nó được điều khiển bởi những `index` nào trong chuỗi trên.
- Nếu ký tự tại `index` đó là 'G', 'g' hoặc 'u', hệ thống cộng thêm `0.5s` vào quỹ thời gian xanh của `index` đó (`link_green_seconds`).

### B. Thời gian xanh hiệu dụng của làn (Lane Green Time)
- Được tính bằng tổng thời gian mà **bất kỳ** dòng nào trên làn đó có đèn xanh. 
- Nếu làn hỗn hợp có hướng đi thẳng xanh 30s, hướng rẽ trái xanh 20s (lệch pha), thì thời gian xanh của làn là thời gian từ lúc hướng đầu tiên xanh đến khi hướng cuối cùng đỏ.
- Giá trị này dùng để tính $f$ (tỷ lệ xanh/chu kỳ) trong công thức tính Delay $t_{w1}$.

---

## 3. Công thức tính toán cuối chu kỳ

Khi kết thúc một chu kỳ (Cycle), hệ thống thực hiện:
1. **Lưu lượng giờ ($q_h$):** `(Tổng PCU thoát / Thời gian chu kỳ) * 3600`.
2. **Hệ số dòng $a_i$:** `PCU thoát của dòng i / Tổng PCU thoát của làn`.
3. **Lưu lượng bão hòa $S_{hh}$:** $S_{hh} = \frac{1}{\sum (a_i / S_i)}$.
4. **Mức độ bão hòa ($x$):** $x = q_h / (S_{hh} \times f)$.

---

## 4. Những điểm bạn có thể cần điều chỉnh (Mapping)
- **Hệ số $S_i$:** Hiện đang tính dựa trên $f_b, f_r, f_d$ chung trong `config_master.json`. Nếu từng hướng rẽ có đặc thù khác nhau, ta có thể tạo một bảng mapping `link_index -> [fb, fr, fd]` riêng.
- **Loại xe:** Đảm bảo tên loại xe trong SUMO khớp với `pcu_mapping`.
