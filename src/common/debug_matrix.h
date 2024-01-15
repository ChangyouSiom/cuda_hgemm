// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: matrix

#pragma once

#include <random>

#include "common.h"

class DebugMatrix {
public:
    DebugMatrix(size_t row, size_t col, const std::string &name = "DebugMatrix")
        : m_row(row), m_col(col), m_name(name) {
        HGEMM_CHECK_GT(m_row, 0);
        HGEMM_CHECK_GT(m_col, 0);

        m_elem_num = m_row * m_col;
        HGEMM_CHECK_GT(m_elem_num, 0);

        m_host_ptr = new int32_t[m_elem_num];
        HGEMM_CHECK(m_host_ptr);
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(int32_t)));
        HGEMM_CHECK(m_dev_ptr);

        for (size_t i = 0; i < m_elem_num; ++i) {
            m_host_ptr[i] = -1;
        }

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(int32_t), cudaMemcpyHostToDevice));

        HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col, m_host_ptr, m_dev_ptr);
    }

    ~DebugMatrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() const {
        return m_row;
    }

    size_t getCol() const {
        return m_col;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    half *getHostPtr() const {
        return m_host_ptr;
    }

    half *getDevPtr() const {
        return m_dev_ptr;
    }

    void tearUp(DebugMatrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        HGEMM_CHECK_CUDART_ERROR(
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    }

    void moveToHost() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    void moveToDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(int32_t));
    }

    void memSetDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(int32_t)));
    }

private:
    const size_t m_row = 0;
    const size_t m_col = 0;
    const std::string m_name = "DebugMatrix";

    size_t m_elem_num = 0;
    half *m_host_ptr = nullptr;
    half *m_dev_ptr = nullptr;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(DebugMatrix);
};
