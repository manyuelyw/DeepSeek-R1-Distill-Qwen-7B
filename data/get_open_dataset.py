from modelscope.msdatasets import MsDataset
ds = MsDataset.load('GAIR/LIMO', subset_name='default', split='train')

# 保存为JSON文件
ds.to_json("limo_dataset.json", orient="records", lines=True)
