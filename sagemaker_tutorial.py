#1
from sagemaker import get_execution_role 
role = get_execution_role() 
bucket = 'gachon-sagemaker-tutorial2' # Use the name of your s3 bucket here 


#2
%%time 
import pickle, gzip, numpy, urllib.request, json 
# Load the dataset 
urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz") 
with gzip.open('mnist.pkl.gz', 'rb') as f: 
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1') 


#3
%matplotlib inline 
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = (2,10) 

def show_digit(img, caption=' ', subplot=None): 
    if subplot == None: 
        _, (subplot) = plt.subplots(1,1) 
    imgr = img.reshape((28,28)) 
    subplot.axis('off') 
    subplot.imshow(imgr, cmap='gray') 
    plt.title(caption) 
    
show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30])) 


#4
from sagemaker import KMeans 
data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket) 
output_location = 's3://{}/kmeans_highlevel_example/output'.format(bucket) 
print('training data will be uploaded to: {}'.format(data_location)) 
print('training artifacts will be uploaded to: {}'.format(output_location)) 
kmeans = KMeans(role=role, # 훈련 결과 읽기 및 쓰기에 사용되는 사용자 IAM 
                train_instance_count=2, # 모델 훈련에 사용할 인스턴스의 수 
                train_instance_type='ml.c4.8xlarge', # 모델 훈련에 사용할 인스턴스의 타입 
                output_path=output_location, # 훈련 결과를 저장할 위치 
                k=10, # 생성할 클러스터의 수, 0부터 9까지의 숫자 분류 문제이기에 10으로 설정 
                data_location=data_location) # 변환된 훈련 데이터를 업로드하는 Amazon S3의 위치


#5
%%time 
kmeans.fit(kmeans.record_set(train_set[0]))  


#6
%%time 
kmeans_predictor = kmeans.deploy(initial_instance_count=1, 
                                instance_type='ml.t2.medium') 

#7
result = kmeans_predictor.predict(valid_set[0][30:31]) 
print(result) 


#8
%%time 
# vaild set에 있는 0번에서 99번까지의 데이터로 클러스터를 예측합니다.
result = kmeans_predictor.predict(valid_set[0][0:100]) 
clusters = [r.label['closest_cluster'].float32_tensor.values[0] for r in result] 
for cluster in range(10): 
    print('\n\n\nCluster {}:'.format(int(cluster))) 
    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ] 
    height = ((len(digits)-1)//5) + 1 
    width = 5 
    plt.rcParams["figure.figsize"] = (width,height) 
    _, subplots = plt.subplots(height, width) 
    subplots = numpy.ndarray.flatten(subplots) 
    for subplot, image in zip(subplots, digits): 
        show_digit(image, subplot=subplot) 
    for subplot in subplots[len(digits):]: 
        subplot.axis('off') 
    plt.show() 
