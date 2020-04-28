import numpy as np


x_1 = np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
x_2 = np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])
y = np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

x11 = []
x22 = []
#数据归一化处理
for i in x_1:
  x_11 = float(i - np.min(x_1))/(np.max(x_1)- np.min(x_1))
  x11.append(x_11)
x1 = np.array(x11)
for i in x_2:
  x_22 = float(i - np.min(x_2))/(np.max(x_2)- np.min(x_2))
  x22.append(x_22)
x2 = np.array(x22)

#计算误差
data = np.stack((x1,x2,y),axis=1)  #转换为numpy数组
print("data=",data)

def Loss(w1,w2,b,data):
    totalError = 0
    for i in range(0,len(data)):
        x1 = data[i,0]
        x2 = data[i,1]
        y = data[i,2]
        totalError += (y-(w1*x1 +w2*x2+b))**2
        return totalError/float((data.size)/2)

#计算梯度

def step_gradient(w1_current,w2_current,b_current,data,lr):
    w1_gradient = 0
    w2_gradient = 0
    b_gradient = 0
    M=16

    for i in range(0,len(data)):
        x1 = data[i,0]
        x2 = data[i,1]
        y = data[i,2]
        w1_gradient += (2 / M) * x1 * ((w1_current * x1 + w2_current * x2 + b_current) - y)
        w2_gradient += (2 / M) * x2 * ((w1_current * x1 + w2_current * x2 + b_current) - y)
        b_current   += (2 / M) * ((w1_current * x1 + w2_current * x2 + b_current) - y)

    new_w1 = w1_current - (lr * w1_gradient)
    new_w2 = w2_current - (lr * w2_gradient)
    new_b = b_current - (lr * b_gradient)
    return [new_w1,new_w2,new_b]

#更新梯度
def gradient_descent(data,star_w1,star_w2,star_b,lr,num_iterations):
    w1 = star_w1
    w2 = star_w2
    b = star_b

    for step in range(num_iterations):
        w1,w2,b = step_gradient(w1,w2,b,data,lr)
        loss = Loss(w1,w2,b,data)
        if step % 5 ==0:
            print(f"iterations:{step}\tloss:{loss}\tw1:{w1}\tw2:{w2}\tb:{b}")
    return[w1,w2,b]

#主函数
def main():
    lr = 0.01
    init_w1 = 0
    init_w2 = 0
    init_b = 0
    num_iterations = 100
    [w1,w2,b] = gradient_descent(data,init_w1,init_w2,init_b,lr,num_iterations)
    loss = Loss(w1,w2,b,data)
    print(f"Final Loss:{loss},w1:{w1},w2:{w2},b:{b}")

main()
















































