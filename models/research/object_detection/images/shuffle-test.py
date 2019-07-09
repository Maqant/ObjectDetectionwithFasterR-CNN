import random
with open('C:\\tensorflow1\\models\\research\\object_detection\\images\\test_labels.csv','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('C:\\tensorflow1\\models\\research\\object_detection\\images\\test_labels.csv','w') as target:
    for _, line in data:
        target.write( line )
		
