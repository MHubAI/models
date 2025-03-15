import os, json, sys


data_file_path = sys.argv[1]

data_dir = os.path.dirname(data_file_path)
nii_file = os.path.basename(data_file_path)


out_json = os.path.join(data_dir,'data.json')

data_json = {
  "val": 
  [
    {
      "image": nii_file
    }
	
  ]
}



json_object = json.dumps(data_json, indent=4)
 
# Writing to sample.json
with open(out_json, "w") as outfile:
    outfile.write(json_object)
