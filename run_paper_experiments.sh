models=("DeepLab" "Unet" "FPN" "TriUnet")

for model in ${models[*]} ; do
  for id in {0..10} ; do
    echo "Training $model $id"
    python train.py "$model" "$id"
  done
done

