e = 2
for i in range(0,2000):
    e = e*0.998
    if(i%100 == 0 and i > 0):
        print(e)

print(e)