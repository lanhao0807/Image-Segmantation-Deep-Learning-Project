import numpy as np
import math

def create_input_affinity_matrix(image, lambda_val = 0.1): 
    # lambda_val is the parameter fixing the balance
    m, n, c = image.shape
    c_norm = image.astype('float') / 255.0
    l_norm = np.zeros((m, n, 2))
    l_norm[:,:,0] = np.arange(m) / (m - 1)
    l_norm[:,:,1] = np.arange(n) / (n - 1)
    f = np.concatenate((c_norm, l_norm), axis=2)
    D = np.zeros((m*n, m*n, 5))
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for t in range(n):
                    index1 = i*n + j
                    index2 = k*n + t
                    D[index1, index2, :] = np.abs(f[i,j,:] - f[k,t,:])
    SrA = np.zeros((m*n, m*n))
    for p in range(m*n):
        for q in range(m*n):
            SrA[p,q] = math.exp(-1 * np.linalg.norm(D[p,q,:3], ord=1) - lambda_val * np.linalg.norm(D[p,q,3:], ord=1))
    return SrA

def create_predicted_mask_affinity_matrix(color_label_map):
    m, n, _ = color_label_map.shape
    mask_vec = color_label_map.reshape(m * n, 3)
    SpA = (mask_vec == mask_vec[:, None, :]).all(axis=2).astype(float)
    return SpA

def split_image_into_blocks(image, N=8):
    m, n, c = image.shape
    m_block_size = m // N
    n_block_size = n // N
    image_blocks = []

    for i in range(0, m, m_block_size):
        row_blocks = []
        for j in range(0, n, n_block_size):
            block = image[i:i + m_block_size, j:j + n_block_size]
            row_blocks.append(block)
        image_blocks.append(row_blocks)

    return image_blocks

def calculate_cosine_similarity(image, mask):
    # Divide the input image and predicted mask into N x N blocks
    image_blocks = split_image_into_blocks(image)
    mask_blocks = split_image_into_blocks(mask)

    # Compute SrA and SpA for each block and calculate cosine similarity
    similarities = []
    for i in range(len(image_blocks)):
        for j in range(len(image_blocks[0])):
            image_block = image_blocks[i][j]
            mask_block = mask_blocks[i][j]
            SrA = create_input_affinity_matrix(image_block)
            SpA = create_predicted_mask_affinity_matrix(mask_block)
            similarity = calculate_cosine_similarity_block(SrA, SpA)
            similarities.append(similarity)

    # Return the 1 - average cosine similarity across all blocks as L term proposed
    return 1 - np.mean(similarities)

def calculate_cosine_similarity_block(SrA, SpA):
    # Compute cosine similarity between SrA and SpA
    Arv_block = SrA.flatten()
    Apv_block = SpA.flatten()
    similarity = np.dot(Arv_block, Apv_block) / (np.linalg.norm(Arv_block) * np.linalg.norm(Apv_block))
    
    # Return the cosine similarity
    return similarity

