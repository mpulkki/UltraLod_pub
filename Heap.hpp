#pragma once

#include <vector>
#include <unordered_map>

namespace UltraLod
{
    template <typename T>
    struct MinHeapComp
    {
        bool operator()(const T& v0, const T& v1) const
        {
            return v0 < v1;
        }
    };

    template <typename TType, typename TComp>
    class HeapHash
    {
    public:
        struct Pair
        {
            TType value;
            int   hash;
        };

    public:
        HeapHash();

        // Returns the root item
        Pair Pop();

        // Pops item with given hash
        Pair Pop(int hash);

        // Pushes a new item to the heap
        bool Push(const TType& value, int hash);

        // Reserves memory for given size
        void Reserve(size_t size);

        // Number of item in the heap
        size_t Size() const;

        void Validate(int idx = 0) const
        {
            if (idx >= Size())
                return;

            auto c0i = ChildIdx(idx, 0);
            auto c1i = ChildIdx(idx, 1);

            if (c0i < Size())
            {
                assert(m_heap[c0i].value >= m_heap[idx].value);
                Validate(c0i);
            }

            if (c1i < Size())
            {
                assert(m_heap[c1i].value >= m_heap[idx].value);
                Validate(c1i);
            }
        }

    private:

        inline int ChildIdx(int idx, int child) const
        {
            assert(child == 0 || child == 1);
            return 2 * idx + child + 1;
        }

        inline bool Compare(int idx0, int idx1) const
        {
            return TComp()(m_heap[idx0].value, m_heap[idx1].value);
        }

        inline int ParentIdx(int idx) const
        {
            return idx > 0 ? (idx - 1) / 2 : -1;
        }

        void HeapifyDown(int idx);
        void HeapifyUp(int idx);

        inline void Swap(int idx0, int idx1)
        {
            assert(idx0 >= 0 && idx0 < Size());
            assert(idx1 >= 0 && idx1 < Size());

            if (idx0 != idx1)
            {
                // Update lookup
                m_lookup[m_heap[idx0].hash] = idx1;
                m_lookup[m_heap[idx1].hash] = idx0;
            }

            // Swap data items
            std::swap(m_heap[idx0], m_heap[idx1]);
        }

    private:
        std::vector<Pair>            m_heap;
        std::unordered_map<int, int> m_lookup;
    };

    template <typename T>
    using MinHeapHash = HeapHash<T, MinHeapComp<T>>;


    // Heap function definitions
    template <typename TType, typename TComp>
    HeapHash<TType, TComp>::HeapHash()
    { }

    template <typename TType, typename TComp>
    void HeapHash<TType, TComp>::HeapifyDown(int idx)
    {
        assert(idx >= 0 && idx < Size());

        while (true)
        {
            auto child0Idx = ChildIdx(idx, 0);
            auto child1Idx = ChildIdx(idx, 1);

            bool c0Ok, c1Ok;

            // Check if child(ren) are violating heap rule
            c0Ok = child0Idx < Size() ? Compare(idx, child0Idx) : true;
            c1Ok = child1Idx < Size() ? Compare(idx, child1Idx) : true;

            if (c0Ok && c1Ok)
                break;

            // Check which child to swap
            int childToSwap = child0Idx;

            if (c0Ok ^ c1Ok)
                childToSwap = !c0Ok ? child0Idx : child1Idx;
            else
                childToSwap = Compare(child0Idx, child1Idx) ? child0Idx : child1Idx;

            // Execute the swap
            Swap(idx, childToSwap);
            idx = childToSwap;
        }
    }

    template <typename TType, typename TComp>
    void HeapHash<TType, TComp>::HeapifyUp(int idx)
    {
        assert(idx >= 0 && idx < Size());

        if (idx == 0)
            return;

        // Bubble new item to top
        for (auto pIdx = ParentIdx(idx); pIdx >= 0; idx = pIdx, pIdx = ParentIdx(pIdx))
        {
            // Check the heap rule
            if (Compare(pIdx, idx))
                break;

            Swap(pIdx, idx);
        }
    }

    template <typename TType, typename TComp>
    typename HeapHash<TType, TComp>::Pair HeapHash<TType, TComp>::Pop()
    {
        assert(Size());
        return Pop(m_heap[0].hash);
    }

    template <typename TType, typename TComp>
    typename HeapHash<TType, TComp>::Pair HeapHash<TType, TComp>::Pop(int hash)
    {
        if (!Size())
            return { TType(), 1 };

        // Validate hash
        auto it = m_lookup.find(hash);

        if (it == m_lookup.end())
            return { TType(), -1 };     // Invalid hash

        auto idx = it->second;
        assert(idx >= 0 && idx < Size());

        // Get value to return
        auto result = m_heap[idx];
        assert(result.hash == hash);

        // Replace with last element
        Swap(idx, (int)m_heap.size() - 1);
        m_heap.pop_back();

        // Remove also lookup
        assert(it->second == Size());        // Must be last element after swap
        m_lookup.erase(it);

        if (idx < Size())
        {
            // Bubble up or down?
            if (idx > 0 && !Compare(ParentIdx(idx), idx))
                HeapifyUp(idx);
            else
                HeapifyDown(idx);
        }

        return result;
    }

    template <typename TType, typename TComp>
    bool HeapHash<TType, TComp>::Push(const TType& value, int hash)
    {
        auto idx = (int)m_heap.size();

        // hash must be unique
        if (m_lookup.find(hash) != m_lookup.end())
            return false;

        // Add item to bottom of the heap
        m_heap.push_back({ value, hash });

        // Add hash to the lookup
        m_lookup.insert({ hash, idx });

        // Move the new item towards top
        HeapifyUp(idx);

        return true;
    }

    template <typename TType, typename TComp>
    void HeapHash<TType, TComp>::Reserve(size_t size)
    {
        m_lookup.reserve(size);
        m_heap.reserve(size);
    }

    template <typename TType, typename TComp>
    size_t HeapHash<TType, TComp>::Size() const
    {
        return m_heap.size();
    }
}