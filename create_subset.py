import os
import shutil
import random

# مسیرهای اصلی (تصاویر اصلی COVID و Normal)
covid_source = './COVID/images'   # مسیر واقعی را اصلاح کنید
normal_source = './Normal/images'

# مسیرهای مقصد برای subset
subset_train_covid = './covid_data_subset/train/covid'
subset_train_normal = './covid_data_subset/train/normal'
subset_test_covid = './covid_data_subset/test/covid'
subset_test_normal = './covid_data_subset/test/normal'

os.makedirs(subset_train_covid, exist_ok=True)
os.makedirs(subset_train_normal, exist_ok=True)
os.makedirs(subset_test_covid, exist_ok=True)
os.makedirs(subset_test_normal, exist_ok=True)

def select_subset(source_dir, train_dest, test_dest, train_size=200, test_size=50):
    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(42)
    selected = random.sample(all_images, train_size + test_size)
    train_files = selected[:train_size]
    test_files = selected[train_size:train_size+test_size]
    
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dest, f))
    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dest, f))
    print(f"{source_dir}: {train_size} to train, {test_size} to test")

select_subset(covid_source, subset_train_covid, subset_test_covid)
select_subset(normal_source, subset_train_normal, subset_test_normal)