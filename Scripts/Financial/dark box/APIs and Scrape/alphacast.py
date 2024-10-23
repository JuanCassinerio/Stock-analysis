#############################################################
'''
https://www.alphacast.io/datasets/monetary-argentina-bcra-seriese-y-bas-selected-daily-5282

'''

from alphacast import Alphacast

key="ak_5HmCDxCdSoYSD22UnXM9"
alphacast = Alphacast(key)
data_code = 5272 #url data code

dataset = alphacast.datasets.dataset(data_code)
df = dataset.download_data(format = "pandas", startDate=None, endDate=None, filterVariables = [], filterEntities = {})