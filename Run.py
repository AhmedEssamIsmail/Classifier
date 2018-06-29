import Main

#Main.gui_intermadiate(1000, "20", 1, "sigmoid", 0.02, 0.005, 2, 1, 1)

testing_path = 'C:/Users/Ahmed/Desktop/Test Cases For Exam'
original_image_path = testing_path + '/' + 'TC2.jpg'
segmented_image_path = testing_path + '/' + 'TC2.png'

Main.test_image(original_image_path, segmented_image_path, 1, "sigmoid")

#Main.RBFrun(10, 0.005, 0.01, 1500)
#Main.test_image_rbf(original_image_path, segmented_image_path)

