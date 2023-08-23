def run_alignment(image_path):
  import dlib
  from scripts.align_all_parallel import align_face
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image

if __name__ == "__main__":
    import os
    os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    os.system("bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2")
    run_alignment(image_path="test.jpg")
    print("Done!")