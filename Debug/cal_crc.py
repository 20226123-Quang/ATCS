import sys
sys.stdout.reconfigure(encoding='utf-8')
def crc16_binary_simulation(bit_string):
    # 1. Kiểm tra đầu vào
    bit_string = bit_string.replace(" ", "") # Xóa khoảng trắng nếu có
    if not all(b in '01' for b in bit_string):
        print("Lỗi: Đầu vào phải là chuỗi nhị phân (0 và 1).")
        return

    #print(f"[*] Dữ liệu gốc: {bit_string}")

    # 2. Đa thức sinh (Generator): x^16 + x^12 + x^2 + 1
    # Tương ứng: 1 0001 0000 0000 0101 (17 bit)
    poly = "10001000000000101"
    poly_len = len(poly) # 17
    crc_len = poly_len - 1 # 16

    # 3. Thêm các bit 0 (Augmenting)
    padded_data = bit_string + ('0' * crc_len)
    remainder = list(padded_data)
    
    #print(f"[*] Đa thức sinh: {poly}")
    #print(f"[*] Dữ liệu sau khi thêm {crc_len} bit 0: {''.join(remainder)}")
    #print("-" * 60)

    # 4. Quá trình chia từng bước
    # Chúng ta chỉ lặp qua chiều dài của chuỗi dữ liệu gốc
    for i in range(len(bit_string)):
        current_bit = remainder[i]
        
        # Hiển thị trạng thái hiện tại
        #print(f"Bước {i+1:02d}: Xét bit tại vị trí {i} (Giá trị: {current_bit})")
        
        if current_bit == '1':
            # Thực hiện phép XOR nếu bit đầu là 1
            #print(f"   Hàng chia: {' ' * i}{''.join(remainder[i:i+poly_len])}")
            #print(f"   Đa thức:  {' ' * i}{poly} (XOR)")
            
            for j in range(poly_len):
                # Phép toán XOR: 1^1=0, 0^0=0, 1^0=1, 0^1=1
                res = int(remainder[i+j]) ^ int(poly[j])
                remainder[i+j] = str(res)
            
            #print(f"   Kết quả:  {' ' * i}{''.join(remainder[i:i+poly_len])}")
        else:
            # Nếu bit đầu là 0, bỏ qua (giống như thương số bằng 0)
            #print(f"   Kết quả:  Giữ nguyên, dịch sang bit tiếp theo.")
        
        #print("." * 40)
            continue
    # 5. Lấy 16 bit cuối cùng làm kết quả
    final_crc = "".join(remainder[-crc_len:])
    
    print("-" * 60)
    #print(f"==> MÃ CRC-16 (Nhị phân): {final_crc}")
    #print(f"==> MÃ CRC-16 (Hex): {hex(int(final_crc, 2)).upper()}")
    return final_crc

# --- CHẠY THỬ ---
# Nhập chuỗi nhị phân của bạn ở đây
input_bits = "00000000 00010011 00000001 00000000 00000001 00000000 00000000 00000000 00000001 00000001 00000001 00010000 00000001 00000001 00000100 00000011 00000011 00000001 00000010" 
crc16_binary_simulation(input_bits)