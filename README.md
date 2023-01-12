## MSDS Capstone Project: Bats!!!

### Raw data storage
[Azure blob storage](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F875b6f94-2db7-46a3-8ecc-1dd2549c188d%2FresourceGroups%2FCapstone_project%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fkirstngcapstone/path/annotated-data/etag/%220x8DAEFA5F2564873%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/Container)

### Progress status

|Stage |Task |Details  |Input |Desired Output |Github Issue |Status |
| --- |:---| :---| :---- |:--- |:---|:---|
|Data cleaning and pre-processing | Develop snipping tool | Developing a snipping tool that allows us to generate images (.png format) that contains bat calls based on the timestamp and frequency parameters from Raven Pro. Eyeballing images is more efficient than loading .wav files into Raven Pro to identify false positive. | 1. Audio files in .wav format.   2.Detection file from Raven Pro in .txt format. | .png images | https://github.com/uw-echospace/bat-detector-msds/issues/9 | In progress |
|Data cleaning and pre-processing | Identify false positive and generate negative sample  | Based on the images snipped from the spectrogram, we will determine the false positive from the generated images. | .png images generated from snipping tool. | .png images that contains bat calls. | https://github.com/uw-echospace/bat-detector-msds/issues/5 | Pending upstream |
|Image clustering | Implement KNN- clustering   | Referring to this [tutorial](https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34) to implement KNN clustering based on visual similarity| filtered .png images containing bat calls. Based on our hypothesis, there should be two distinct bat calls, ie feeding buzz and non-feeding buzz. | Images that are in each group. | TBD| Pending upstream |


