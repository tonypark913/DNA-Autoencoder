"""
dataProcessing.py
Author: 박지민 (tonypark913)
Email: tonypark913@naver.com
Date: 2023-11-21

이 모듈은 DNA 데이터를 처리하는 다양한 함수를 포함하고 있습니다. 
함수들은 DNA 데이터를 다운로드하고, 데이터에 zero padding을 적용하며, DNA 시퀀스를 숫자로 변환하고, 
DNA 시퀀스를 확장하며, 가우시안 필터를 생성하고, DNA 시퀀스에 가우시안 필터를 적용하며, 
DNA 데이터를 처리하고, CSV 파일에서 DNA 데이터를 읽어오는 기능을 수행합니다.

이 모듈은 csv, os, http.client, time, numpy, pandas, tqdm, scipy.stats, Bio, log 등의 
다양한 Python 라이브러리와 모듈을 사용합니다.
"""


import csv  # CSV 파일 처리를 위한 모듈
import os  # 운영체제와 상호작용하기 위한 모듈
import http.client  # HTTP 프로토콜 클라이언트
import time  # 시간 관련 함수
import numpy as np  # 배열 및 행렬 연산을 위한 라이브러리
import pandas as pd  # 데이터 분석 라이브러리
import ast  # 문자열을 파싱하기 위한 라이브러리
from tqdm import tqdm  # 진행 표시줄 라이브러리
from scipy.stats import norm  # 통계 관련 라이브러리
from Bio import Entrez, SeqIO  # 생물정보학 라이브러리
from log import logger  # 로깅 모듈

logger.info("dataProcessing imported.")

# DNA 시퀀스를 다루기 위한 변수들
permutationACTG=[['A', '', '', ''], ['A', 'C', 'T', 'G'], ['A', 'C', 'G', 'T'], ['A', 'T', 'C', 'G'], ['A', 'T', 'G', 'C'], ['A', 'G', 'T', 'C'], ['A', 'G', 'C', 'T'],
                 ['C', '', '', ''], ['C', 'A', 'T', 'G'], ['C', 'A', 'G', 'T'], ['C', 'G', 'A', 'T'], ['C', 'G', 'T', 'A'], ['C', 'T', 'A', 'G'], ['C', 'T', 'G', 'A'],
                 ['T', '', '', ''], ['T', 'C', 'A', 'G'], ['T', 'C', 'G', 'A'], ['T', 'G', 'A', 'C'], ['T', 'G', 'C', 'A'], ['T', 'A', 'C', 'G'], ['T', 'A', 'G', 'C'],
                 ['G', '', '', ''], ['G', 'C', 'T', 'A'], ['G', 'C', 'A', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'T', 'C'], ['G', 'T', 'C', 'A'], ['A', 'T', 'A', 'C']]

oneHotEncodingBase=['A', 'C', 'T', 'G']  # 원-핫 인코딩을 위한 베이스

normbase='ATGTCGCGCTTGGCACGCACGAGCCGCCCGCCGCGTCCCCCTGGCTCCGGGGCCAGCCGAGACCTGCGGCCGCCCGGGGCGCAGTCAACCCGCCCCCCGC'  # 정규화를 위한 베이스

# DNA 데이터를 다운로드하는 함수

# downloadDnaData(query: str, size: int = 600, filename: str = 'sampleData') -> None: 
# 이 함수는 DNA 데이터를 다운로드하는 함수입니다. 
# query는 검색할 DNA 시퀀스, size는 검색 결과로 가져올 데이터의 크기, filename은 다운로드한 데이터를 저장할 파일의 이름을 나타냅니다.

def downloadDnaData(query: str, size: int = 600, filename: str = 'sampleData') -> None: #데이터 다운로드
    logger.info("dataProcessing-downloadDnaData: started.")  # 로깅 시작
    log_filename = f"{filename}_log.csv"  # 로그 파일 이름 설정
    start_time = time.time()  # 시작 시간 기록
    Entrez.email = "jshs20221413@h.jne.go.kr"  # Entrez에 이메일 설정
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=size*3)  # Entrez를 이용해 nucleotide 데이터베이스 검색
    record = Entrez.read(handle)  # 결과 읽기
    count = int(record["Count"])  # 결과 개수
    logger.info(f"dataProcessing-downloadDnaData: Query: {query} found count: {count} genes.")  # 로그에 결과 개수 기록
    if size>count:
      logger.warning(f"dataProcessing-downloadDnaData: size: {size} is larger than query count: {int(count)}")  # 요청 크기가 결과 개수보다 클 경우 경고 로그 작성
      return None
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=size*3)  # 다시 검색
    record = Entrez.read(handle)  # 결과 읽기
    id_list = record["IdList"]  # ID 리스트 가져오기 
    max_records=size  # 최대 레코드 수 설정
    processed_records = 0  # 처리된 레코드 수 초기화
    logger.info(f"dataProcessing-downloadDnaData: download query: {query} size: {size}")  # 로그에 다운로드 정보 기록
    pbar = tqdm(total=max_records, desc='Downloading DNA Data', unit='record')  # 진행 표시줄 설정
    with open(filename, mode='a', newline='') as file:  # 파일 열기
        writer = csv.writer(file)  # CSV 작성자 생성
        # 파일이 처음 생성되는 경우에만 헤더 작성
        if os.stat(filename).st_size == 0:
          writer.writerow(["Sequence"])
        with open(log_filename, mode='a', newline='') as log_file:  # 로그 파일 열기
            log_writer=csv.writer(log_file)  # 로그 작성자 생성
             # 로그 파일이 처음 생성되는 경우에만 헤더 작성
            if os.stat(log_filename).st_size == 0:
                 log_writer.writerow(["Query", "Size"])
            log_writer.writerow([query, size])  # 로그에 쿼리와 크기 기록
        retstart=0  # Start index for esearch query
        last_description_part = None
        last_seq_100bp = None
        while processed_records < max_records:  # 처리된 레코드 수가 최대 레코드 수보다 작은 동안 반복
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=1000, retstart=retstart)  # Entrez를 이용해 nucleotide 데이터베이스 검색
            record = Entrez.read(handle)  # 결과 읽기
            id_list=record["IdList"]  # ID 리스트 가져오기
            for i in range(0, len(id_list), 100):  # ID 리스트를 100개씩 처리
                if processed_records >= max_records:  # 처리된 레코드 수가 최대 레코드 수보다 크거나 같으면 종료
                    break
                batch_id_list=id_list[i:i+100]  # 100개의 ID를 가져옴
                for _ in range(5):  # 최대 5번 재시도
                    try:
                        handle2_batch=Entrez.efetch(db="nucleotide", id=batch_id_list, rettype="gb", retmode="text")  # 각 ID에 대한 정보를 가져옴
                        break  
                    except (Exception, http.client.IncompleteRead) as e:  # 오류 발생 시 재시도
                        logger.warning(f"dataProcessing-downloadDnaData: Error occurred: {e}. Retrying...")
                        continue
                records_batch=SeqIO.parse(handle2_batch,"genbank")  # 가져온 정보를 파싱
                for record2 in records_batch:  # 각 레코드에 대해
                    try: 
                        if processed_records < max_records and len(record2.seq) >= 1 and len(record2.seq) <= 50000 and len(record2.seq)>=20:  # 레코드의 시퀀스 길이가 적절한 경우
                            current_description_part = record2.description.split(", transcript variant")[0]  # 설명 부분을 파싱
                            if current_description_part == last_description_part:  # 이전 설명과 같으면 건너뜀
                               last_description_part = current_description_part
                               continue
                            current_sequence_100bp=str(record2.seq[101:200])  # 시퀀스의 처음 100bp를 가져옴
                            if current_sequence_100bp == last_seq_100bp:  # 이전 시퀀스와 같으면 건너뜀
                                last_seq_100bp = current_sequence_100bp
                                continue
                            last_seq_100bp = current_sequence_100bp  # 이전 시퀀스를 현재 시퀀스로 업데이트
                            last_description_part = current_description_part  # 이전 설명을 현재 설명으로 업데이트
                            writer.writerow([current_sequence_100bp])  # 시퀀스를 파일에 씀
                            processed_records +=1  # 처리된 레코드 수 증가
                            pbar.update(1)  # 진행 표시줄 업데이트
                    except Exception as e:  # 오류 발생 시 건너뜀
                        continue
            retstart += len(id_list)  # 다음 검색을 위해 시작 인덱스를 업데이트
        pbar.close()  # 진행 표시줄 닫기
    end_time = time.time()  # 종료 시간 기록
    execution_time = end_time - start_time  # 실행 시간 계산
    logger.info(f"dataProcessing-downloadDnaData: Function executed in: {execution_time:.1f} seconds")  # 로그에 실행 시간 기록
    logger.info(f"dataProcessing-Download finished for query: {query} in records: {processed_records}")  # 로그에 다운로드 완료 정보 기록
    handle.close()  # 핸들 닫기

# dataZeropadding(targetSequence: list, normSequence: str) -> list:
# 이 함수는 DNA 시퀀스에 zero padding을 적용하는 함수입니다.
# targetSequence는 padding을 적용할 DNA 시퀀스, normSequence는 정규화를 위한 DNA 시퀀스를 나타냅니다.

def dataZeropadding(targetSequence: str, normSequence: str) -> str:  # targetSequence의 길이를 normSequence의 길이에 맞게 0 패딩
    # logger.info(f"dataProcessing-dataZeropadding: started.")
    if len(targetSequence) >= len(normSequence):  # targetSequence의 길이가 normSequence의 길이보다 크거나 같으면 오류 발생
        raise ValueError(f"dataProcessing-dataZeropadding: targetSequence: {len(targetSequence)} is larger than(or equal) normSeqeunce: {len(normSequence)}.")
    else:
        # Add 'Z' to the end of target sequence until its length matches with norm sequence
        padded_seq = targetSequence + 'Z' * (len(normSequence) - len(targetSequence))  # targetSequence의 끝에 'Z'를 추가하여 길이를 맞춤
        # logger.info(f"dataProcessing-dataZeropadding: finished for targerSequence: {len(targetSequence)} and normSequence: {len(normSequence)} with return seq: {len(padded_seq)}")
        return padded_seq

def dataReduction(targetSequence: str, normSequence: str) -> str:  # targetSequence의 길이를 normSequence의 길이에 맞게 축소
    # logger.info(f"dataProcessing-dataReduction: started.")
    if len(targetSequence) <= len(normSequence):  # targetSequence의 길이가 normSequence의 길이보다 작거나 같으면 오류 발생
        raise ValueError(f"dataProcessing-dataReduction: targetSequence: {len(targetSequence)} is smaller than(or equal) normSeqeunce: {len(normSequence)}.")
    else:
        # Cut the end of target sequence until its length matches with norm sequence
        reduced_seq = targetSequence[:len(normSequence)]  # targetSequence의 끝을 잘라서 normSequence의 길이와 같게 만듦
        # logger.info(f"dataProcessing-dataReduction: finished for targetSequence: {len(targetSequence)} and normSequence: {len(normSequence)} with return seq: {len(reduced_seq)}")
    return reduced_seq

# data2num(targetSequence: list, N1: str, N2: str, N3: str, N4: str) -> list:
# 이 함수는 DNA 시퀀스를 숫자로 변환하는 함수입니다.
# targetSequence는 변환할 DNA 시퀀스, N1, N2, N3, N4는 각각 DNA 시퀀스의 'A', 'C', 'T', 'G'를 대응하는 숫자로 변환하는데 사용됩니다.

def data2num(targetSequence:list, N1:str, N2:str, N3:str, N4:str)->list: #데이터 숫자로 변환
    # logger.info(f"dataProcessing-data2num started.") 
    numGene=[]
    for i in range(len(targetSequence[0])):
        if str(targetSequence[0][i]) == 'U':  # 'U'는 'T'로 변환
            targetSequence[0][i] = 'T'
        # 각 유전자에 대해 해당하는 숫자로 변환    
        if str(targetSequence[0][i]) == N1:
            numGene.append(1)
        elif str(targetSequence[0][i]) == N2:
            numGene.append(0.75)
        elif str(targetSequence[0][i]) == N3:
            numGene.append(0.5)
        elif str(targetSequence[0][i]) == N4:
            numGene.append(0.25)
        elif str(targetSequence[0][i]) == 'N':
            numGene.append(0.125)
        elif str(targetSequence[0][i]) == 'Z':
            numGene.append(0.05)
        elif str(targetSequence[0][i]) == 'A' or str(targetSequence[0][i]) == 'C' or str(targetSequence[0][i]) == 'T' or str(targetSequence[0][i]) == 'G':
            numGene.append(0)
        else:
            numGene.append(0)
    # logger.info(f"dataProcessing-data2num: finished for targetSequence: {len(targetSequence)} with base N1(0.25): {N1}, N2(0.5): {N2}, N3(0.75): {N3}, N4(1): {N4}, N5: N")
    return numGene

# dataAppend(targetSequence: list) -> list:
# 이 함수는 DNA 시퀀스를 확장하는 함수입니다.
# targetSequence는 확장할 DNA 시퀀스를 나타냅니다.

def dataAppend(targetSequence:list)->list: #데이터 확대(1차원->28차원)
    # logger.info(f"dataProcessing-dataAppend: started.")
    newSequence=[]
    for i in range(28):
        newSequence.append(data2num(targetSequence, permutationACTG[i][0], permutationACTG[i][1], permutationACTG[i][2], permutationACTG[i][3]))
    # logger.info(f"dataProcessing-dataAppend: finished for targetSequence: {len(targetSequence)} with base permutation 24 and oneHotEncoding 4.")
    return newSequence

# gaussian_filter(size: int = 5, sigma: float = 1.0) -> list:
# 이 함수는 가우시안 필터를 생성하는 함수입니다.
# size는 필터의 크기, sigma는 가우시안 분포의 표준편차를 나타냅니다.

def gaussian_filter(size: int=5, sigma: float=1.0)->list: #가우시안 필터 생성
    # logger.info(f"dataProcessing-gaussian_filter: started.")
    if size%2==0:
        logger.error(f"dataProcessing-gaussian_filter: size: {size} must be odd.")
    else:
        x = np.linspace(-size // 2, size // 2, size)
        filter = norm.pdf(x, 0, sigma)
        filter /= np.sum(filter)
    #    logger.info(f"dataProcessing-gaussian_filter: finished.")
        return filter

# convolutionGaussianFilter(targetSequence: list, filter: list) -> list:
# 이 함수는 DNA 시퀀스에 가우시안 필터를 적용하는 함수입니다.
# targetSequence는 필터를 적용할 DNA 시퀀스, filter는 적용할 가우시안 필터를 나타냅니다.

def convolutionGaussianFilter(targetSequence: list, filter: list)->list: #가우시안 필터와 컨볼루션
    # logger.info(f"dataProcessing-convolutionGaussianFilter: started.")
    filter_size = len(filter)
    pad_size = filter_size // 2
    padded_array = np.pad(targetSequence, pad_width=(pad_size,), mode='constant', constant_values=0)  # 패딩 추가
    convoluted = []
    
    for i in range(pad_size, len(padded_array) - pad_size):  # 패딩된 배열을 순회
        s = sum(padded_array[i - pad_size: i + pad_size + 1][j] * filter[j] for j in range(filter_size))  # 가우시안 필터를 적용하여 컨볼루션 계산
        convoluted.append(s)  # 결과를 convoluted 리스트에 추가
    
    # logger.info(f"dataProcessing-convolutionGaussianFilter: finished.")
    return convoluted  # 컨볼루션된 리스트 반환

# dataProcess(targetData: list, filter: list) -> list: 
# 이 함수는 DNA 데이터를 처리하는 함수입니다.
# targetData는 처리할 DNA 데이터, filter는 데이터 처리에 사용할 가우시안 필터를 나타냅니다.

def dataProcess(targetData: list, filter: list)->list:  # 데이터 처리 함수
    logger.info(f"dataProcessing-dataProcess: started.")
    newData=[]

    # pbar = tqdm(range(len(targetData)), desc='Data Processed', unit='data')  # 진행 상황을 표시할 progress bar
    # for data in pbar:  # targetData를 순회
    mData1=[]
    targetData_1=[]
    if(len(targetData)<100):  # 데이터의 길이가 100보다 작은 경우
        targetData_1.append(dataZeropadding(targetData, normbase))  # 데이터에 zero padding 적용
        mData=dataAppend(targetData_1)  # 데이터 확장
        for j in range(28):
            mData1.append(convolutionGaussianFilter(mData[j], filter))  # 가우시안 필터 적용
    else:  # 데이터의 길이가 100 이상인 경우
        mData=dataAppend(targetData)  # 데이터 확장
        for j in range(28):
            mData1.append(convolutionGaussianFilter(mData[j], filter))  # 가우시안 필터 적용
    newData.append(mData1)  # 처리된 데이터를 newData에 추가
    
    logger.info(f"dataProcessing-dataProcess: finished with data : {len(targetData[0])}.")
    return newData  # 처리된 데이터 반환

# dataAccess(fileName: str = 'sampleData', start: int = 0, num: int = 100, randopt: bin = 0) -> list:
# 이 함수는 CSV 파일에서 DNA 데이터를 읽어오는 함수입니다.
# fileName은 데이터를 읽어올 파일의 이름, start는 읽기 시작할 행의 인덱스, num은 읽어올 행의 수, randopt는 무작위로 행을 선택할지 여부를 나타냅니다.

def dataAccess(fileName: str='sampleData', start: int=0, num: int=100, randopt: bin=0)->list:  # 데이터 접근 함수
    logger.info(f"dataPrcessing-dataAccess: started.")
    
    chunksize = 10 ** 3 # 한 번에 읽어올 데이터의 크기
    chunks = []  # 데이터를 저장할 리스트
    
    try:
        if randopt == 1:  # 무작위 선택 옵션인 경우
            df = pd.read_csv(fileName)  # 파일에서 모든 데이터를 읽어옴
            logger.info(f"dataPrcessing-dataAccess: finished with data num : {num} from data : {fileName}.")  # 무작위로 num개의 데이터를 선택하여 반환
            return df.sample(n=num).values
        else:
            row_count = 0
            for chunk in pd.read_csv(fileName, chunksize=chunksize):  # 파일에서 데이터를 chunksize만큼씩 읽어옴
                if row_count < start + num:  # 아직 필요한 만큼의 데이터를 읽지 못한 경우
                    chunks.append(chunk.iloc[max(0, start-row_count):min(start-row_count+num, len(chunk))])  # 필요한 만큼의 행만 선택하여 추가
                    row_count += len(chunk)
                else:  # 필요한 만큼의 데이터를 모두 읽은 경우
                    logger.info(f"dataPrcessing-dataAccess: finished with data num : {num} from data : {fileName}.")
                    break

        # 모든 청크들을 합친 후 numpy 배열로 변환하여 반환
        
        return pd.concat(chunks).values 
    except FileNotFoundError:
        logger.error(f"dataPrcessing-dataAccess: Error: File {fileName} not found")
        return None

def devide_train_test_dataset(total_data: int=272443, test_data: int=30000, file: str='processedData.csv') -> None:
    # Read the data from the file
    data = pd.read_csv(file)
    
    # Randomly select test_data number of rows for the test dataset
    test_dataset = data.sample(n=test_data)
    
    # Remove the selected rows from the original dataset to create the train dataset
    train_dataset = data.drop(test_dataset.index)
    
    # Save the test dataset to test_data.csv
    test_dataset.to_csv('test_data.csv', index=False)
    
    # Save the train dataset to train_data.csv
    train_dataset.to_csv('train_data.csv', index=False)

def csv_to_arrays(PATH: str="processedData/train_dataset/processedData_0.csv") -> list: # CSV 파일을 읽어서 numpy 배열로 변환하는 함수
    # CSV 파일 읽기
    df = pd.read_csv(PATH)

    # 모든 행을 가져와서 각 행을 28x28 배열로 변환
    arrays = []
    for _, row in df.iterrows():
        array = []
        for item in row:
            # 각 행을 파싱하고, numpy 배열로 변환
            array.append(ast.literal_eval(item))
        # 배열을 numpy 배열로 변환
        array = np.array(array)
        arrays.append(array)

    # 모든 배열을 하나의 numpy 배열로 변환
    arrays = np.array(arrays)

    return arrays
