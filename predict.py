import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import os
import argparse

# Định nghĩa ArgumentParser để lấy các tham số từ dòng lệnh
def parse_args():
    parser = argparse.ArgumentParser(description="Train a football match prediction model.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Số epoch cho quá trình huấn luyện")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate cho optimizer")
    parser.add_argument("--trained_models_dir", type=str, default='./Final_model', help="Thư mục lưu các mô hình đã huấn luyện")
    parser.add_argument("--checkpoint_path", type=str, default='./Final_model/checkpoint_epoch_29.pth', help="Đường dẫn tới checkpoint để tiếp tục huấn luyện")
    
    args = parser.parse_args()
    return args

# Lấy các đối số từ dòng lệnh
args = parse_args()

# Set display option to show all columns
pd.set_option('display.max_columns', None)
# Danh sách lưu các DataFrame
dfs = []
folder_path = r"C:\Users\ADMIN\Desktop\DEEP LEARNING MATERIAL\WEB SCRAPING"

# Lặp qua tất cả các file trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Kiểm tra nếu là file CSV
        file_path = os.path.join(folder_path, filename)

        # Đọc file CSV mà không để cột 'date' làm chỉ mục (index)
        df = pd.read_csv(file_path)

        # In ra danh sách cột
        print(f"Columns in {filename}: {df.columns}")
        
        # Thêm vào danh sách dfs
        dfs.append(df)

# Nối tất cả các DataFrame lại với nhau
maches = pd.concat(dfs, ignore_index=True)

# Chuyển index thành một cột bình thường
maches = maches.reset_index(drop=True)

maches.drop(['match report','comp'],axis=1,inplace=True)
maches.drop(['notes','attendance','captain','season'],axis=1,inplace=True)

# 1. Tách cột 'date' thành các thành phần ngày, tháng, năm
maches['date'] = pd.to_datetime(maches['date'])  # Chuyển cột 'date' thành datetime

# Tạo các cột 'day', 'month', 'year' từ cột 'date'
maches['day'] = maches['date'].dt.day  # Lấy ngày
maches['month'] = maches['date'].dt.month  # Lấy tháng
maches['year'] = maches['date'].dt.year  # Lấy năm

# # Tách giờ địa phương và giờ chuẩn
# # Cột 'time' chứa giờ địa phương và giờ chuẩn, ví dụ: "12:30 (18:30)"
maches['local_time'] = maches['time'].str.split('(', expand=True)[0].str.strip()  # Lấy giờ địa phương trước dấu '('
# maches['gmt_time'] = maches['time'].str.extract(r'\((.*?)\)', expand=False)  # Lấy giờ chuẩn (GMT) trong dấu '('

# Tạo các cột 'local_hour' và 'local_minute' cho giờ và phút địa phương
maches['local_hour'] = maches['local_time'].str.split(':').str[0].astype(int)  # Lấy giờ từ giờ địa phương
maches['local_minute'] = maches['local_time'].str.split(':').str[1].astype(int)  # Lấy phút từ giờ địa phương

maches.drop(['date','time','local_time'],axis=1,inplace=True)

# Hiển thị kết quả sau khi tách và chuyển đổi
maches[[ 'day', 'month', 'year']].head()
# Bước 1: Xử lý cột phân loại (One-Hot Encoding cho các cột phân loại)
categorical_columns = ['round', 'venue', 'result', 'opponent', 'formation', 'opp formation', 'referee', 'team']

# Bước 2: Xử lý cột số (chuẩn hóa và xử lý giá trị thiếu)
numeric_columns = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'month', 'year', 'local_hour', 'local_minute']

# Bước 3: Chuẩn bị bộ dữ liệu cho huấn luyện
X = maches[categorical_columns + numeric_columns ]
y = maches['result'].map({'W': 1, 'L': 0, 'D': 2})  # Mã hóa kết quả trận đấu

# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Bước 7: Tạo pipeline cho các bước tiền xử lý và huấn luyện mô hình
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Xử lý giá trị thiếu cho cột số
            ('scaler', StandardScaler())  # Chuẩn hóa các cột số
        ]), numeric_columns ),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Xử lý giá trị thiếu cho cột phân loại
            ('encoder', OneHotEncoder(sparse_output=False))  # One-hot encoding cho các cột phân loại
        ]), categorical_columns)
    ]
)
# Kiểm tra các cột không phải số và chuyển chúng thành giá trị số
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Chuyển đổi dữ liệu từ DataFrame thành NumPy array và sau đó thành tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Chuyển đổi nhãn mục tiêu (y_train, y_test) thành tensor
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
# Kiểm tra nếu GPU có sẵn và chọn device (GPU nếu có, CPU nếu không)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Xây dựng mô hình Neural Network
class FootballModel(nn.Module):
    def __init__(self, input_dim):
        super(FootballModel, self).__init__()
        # Định nghĩa các lớp ẩn và đầu ra
        self.fc1 = nn.Linear(input_dim, 64)  # Lớp ẩn với 64 neuron
        self.fc2 = nn.Linear(64, 32)  # Lớp ẩn tiếp theo với 32 neuron
        self.fc3 = nn.Linear(32, 3)  # Lớp đầu ra với 3 neuron (vì có 3 lớp: Win, Draw, Loss)
        self.relu = nn.ReLU()  # Hàm kích hoạt ReLU
        self.softmax = nn.Softmax(dim=1)  # Hàm softmax cho đầu ra phân loại

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Lớp ẩn đầu tiên
        x = self.relu(self.fc2(x))  # Lớp ẩn thứ hai
        x = self.fc3(x)  # Lớp đầu ra
        return self.softmax(x)  # Xác suất cho mỗi lớp

# Khởi tạo mô hình
model = FootballModel(input_dim=X_train_tensor.shape[1])
# Đưa mô hình vào đúng thiết bị (GPU hoặc CPU)
model = model.to(device)

# Định nghĩa optimizer và loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Huấn luyện mô hình
num_epochs = 100
best_accuracy = 0.0  # Để theo dõi mô hình tốt nhất
best_model_path = "./Final_model/best_model.pth"  # Đường dẫn lưu mô hình tốt nhất
# Đảm bảo bạn có thể tạo thư mục để lưu mô hình
import os
trained_models_dir = './Final_model'
os.makedirs(trained_models_dir, exist_ok=True)
# Sử dụng tqdm để theo dõi tiến trình huấn luyện
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    optimizer.zero_grad()  # Đặt lại gradient về 0

    # Đưa dữ liệu vào GPU hoặc CPU
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

    # Tiến hành forward pass
    outputs = model(X_train_tensor)
    
    # Tính toán loss
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass và cập nhật trọng số
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Đánh giá mô hình trên tập kiểm tra
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)

        y_pred = model(X_test_tensor)
        _, predicted = torch.max(y_pred, 1)  # Lấy lớp có xác suất cao nhất
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

        # Lưu checkpoint sau mỗi epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "accuracy": accuracy
        }
        checkpoint_path = f"{trained_models_dir}/last_cnn_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch+1} with accuracy: {accuracy:.2f}")

        # Lưu mô hình tốt nhất nếu accuracy cao hơn
        if accuracy > best_accuracy:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": accuracy,   
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            best_model_path = f"{trained_models_dir}/best_cnn_epoch_{epoch+1}.pt"
            torch.save(checkpoint, best_model_path)
            best_accuracy = accuracy
            print(f"Best model saved for epoch {epoch+1} with accuracy: {accuracy:.2f}")

        # In báo cáo phân loại chi tiết
        print(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}")
        report = classification_report(y_test_tensor.cpu(), predicted.cpu())
        print('Classification Report:\n', report)

        # Tính toán và in ma trận nhầm lẫn
        cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
        print('Confusion Matrix:\n', cm)

        # Tính và in F1-score
        f1 = f1_score(y_test_tensor.cpu(), predicted.cpu(), average='weighted')  # Hoặc 'macro' hoặc 'weighted' cho bài toán đa lớp
        print(f'F1 Score: {f1:.4f}')