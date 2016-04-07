import os
trainxdirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_HE/Train/'
trainydirname = '/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_HE/Train_anno/'

included_extensions = ['mat']
trainfilelist = [fn[:-6] for fn in os.listdir(trainxdirname) if any([fn.endswith(ext) for ext in included_extensions])]
trainfilelist = list(set(trainfilelist))

included_extensions = ['bmp']
train_annofilelist = [fn[:-8] for fn in os.listdir(trainydirname) if any([fn.endswith(ext) for ext in included_extensions])]
train_annofilelist = list(set(train_annofilelist))

diff = set(trainfilelist) - set(train_annofilelist)
print diff

revdiff = set(train_annofilelist) - set(trainfilelist)
print revdiff
