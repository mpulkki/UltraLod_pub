#pragma once

#include <assert.h>
#include <utility>

namespace UltraLod
{
    class Sorting
    {
    public:

        // Quicksort with key and value pairs
        template <typename TType, typename TValue>
        static void QuickSort(TType* ptr, TValue* values, int len)
        {
            if (!ptr || !values || !len)
                return;

            QuickSort(ptr, values, 0, len);
        }


    private:

        template <typename TType, typename TValue>
        static inline int Partition(TType* ptr, TValue* values, int lo, int hi)
        {
            auto pIdx   = lo;
            auto pValue = ptr[pIdx];

            for (auto i = lo + 1; i < hi; i++)
            {
                if (ptr[i] <= pValue)
                {
                    pIdx++;
                    std::swap(ptr[i], ptr[pIdx]);
                    std::swap(values[i], values[pIdx]);
                }
            }

            std::swap(ptr[pIdx], ptr[lo]);
            std::swap(values[pIdx], values[lo]);

            return pIdx;
        }

        template <typename TType, typename TValue>
        static inline void QuickSort(TType* ptr, TValue* values, int lo, int hi)
        {
            if (lo < hi)
            {
                auto pivot = Partition(ptr, values, lo, hi);
                QuickSort(ptr, values, lo, pivot);
                QuickSort(ptr, values, pivot + 1, hi);
            }
        }
    };
}