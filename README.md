# ai-bg-removal
remove background using model from huggingface

## Building

```
docker build . -t bg-removal:0.0.1
```

## Running

```
mkdir -p ./input ./output
docker run --rm -p 5000:5000 -v ./input:/app/input -v ./output:/app/output bg-removal:0.0.1
```

## Using

Leave the .jpg files in the ./input dir, run the app, click run, check the files in the output dir.
