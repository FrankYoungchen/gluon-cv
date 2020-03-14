import os
otb50 = ['Basketball','Biker','Bird1','BlurBody','BlurCar2','BlurFace','BlurOwl',
'Bolt','Box','Car1','Car4','CarDark','CarScale','ClifBar','Couple','Crowds','David','Deer',
'Diving','DragonBaby','Dudek','Football','Freeman4','Girl','Human3','Human4','Human6','Human9',
'Ironman','Jump','Jumping','Liquor','Matrix','MotorRolling','Panda','RedTeam','Shaking','Singer2',
'Skating1','Skating2','Skiing','Soccer','Surfer','Sylvester','Tiger2','Trellis','Walking',
'Walking2','Woman']
otb100 = ['Bird2','BlurCar1','BlurCar3','BlurCar4','Board','Bolt2','Boy',
'Car2','Car24','Coke','Coupon','Crossing','Dancer','Dancer2','David2',
'David3','Dog','Dog1','Doll','FaceOcc1','FaceOcc2','Fish','FleetFace','Football1',
'Freeman1','Freeman3','Girl2','Gym','Human2','Human5','Human7','Human8','Jogging',
'KiteSurf','Lemming','Man','Mhyang','MountainBike','Rubik','Singer1','Skater',
'Skater2','Subway','Suv','Tiger1','Toy','Trans','Twinnings','Vase']
base = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/'
for per in otb50:
    os.system('wget '+base+''+per+'.zip')
    os.system('unzip '+per+'.zip'+' -d ./')
for per in otb100:
    os.system('wget '+base+''+per+'.zip')
    os.system('unzip '+per+'.zip'+' -d ./')

os.system('cp -r Jogging Jogging-1')
os.system('cp -r Jogging Jogging-2')
os.system('cp -r Skating2 Skating2-1')
os.system('cp -r Skating2 Skating2-2')
os.system('cp -r Human4 Human4-2')

