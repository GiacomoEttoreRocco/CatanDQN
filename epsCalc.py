e = 1
for i in range(0,3000):
    e = e*0.999
    if(i%100 == 0 and i > 0):
        print(e)

print(e)