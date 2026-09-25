"""
fair_queue.py — 单文件有界阻塞多生产者多消费者队列（仅标准库）。

设计要点
========
存储
    定长 list 作环形缓冲（head 指针 + count），容量固定，O(1) 入队/出队。
    不依赖 queue.Queue / collections.deque。

同步
    一把互斥锁 + 两个条件变量（not_empty / not_full），经典 MPMC 结构。
    put/take/take_all 的"检查-修改"全程在锁内完成，因此每个元素
    恰好被一个消费者取走（count/head 的修改是原子的，不存在两个
    消费者读到同一槽位的窗口）。

公平性 / 无饿死
    状态变化时用 notify_all() 而非 notify(1)：
      1) 不会丢唤醒——所有等待者都会重新检查条件，不存在"信号被
         错误的线程吸收"的问题（notify(1) 在多消费者竞争下可能把
         唯一信号交给一个随即发现条件已不成立的线程）；
      2) 等待者被唤醒后按"先等待先竞争锁"的顺序排队（CPython 的
         Condition 内部把被通知者按序移入外层锁的等待队列），
         配合 while 循环重检，任何等待者在条件反复成立时必然最终
         抢到锁——队列有界保证生产者不能无限甩开消费者，因此等待
         的消费者被唤醒的次数有限次内必然成功，不会永久饿死。
    代价是惊群（thundering herd）：每次唤醒 O(等待者数)。容量和
    线程数不大时这是正确性/公平性最稳的取舍；若追求极致吞吐可改
    为每线程独立 Condition 的显式 FIFO 等待队列，但复杂度大增。

队满策略（构造时可配置）
    put_block=True（默认）：put 阻塞直到有空位或队列关闭。
        取舍：天然背压（backpressure），生产者不会丢任务也不会撑爆
        内存，适合"任务不能丢"的场景；代价是生产者可能被拖慢。
    put_block=False：队满时 put 立刻（或在 put_timeout超时后）抛
        QueueFull。取舍：生产者永不被卡死，可自行决定丢弃/重试/
        落盘，适合实时性优先、可容忍丢任务的场景；代价是调用方
        要处理失败路径。
    单次调用也可用 put(item, block=..., timeout=...) 覆盖默认值。

优雅关闭
    close() 后：
      - 新 put 立即抛 QueueClosed（阻塞中的 put 也被唤醒并抛出）；
      - 已入队的任务照常可被消费；
      - 队列空且已关闭时，take/take_all 返回结束信号 END。
"""

from __future__ import annotations

import threading
import time

__all__ = [
    "BoundedMPMCQueue",
    "QueueClosed",
    "QueueFull",
    "QueueEmpty",
    "END",
]


class QueueClosed(Exception):
    """队列已关闭：put 被拒绝，或阻塞中的 put 被关闭打断。"""


class QueueFull(Exception):
    """非阻塞/超时模式下队满，put 失败。"""


class QueueEmpty(Exception):
    """非阻塞/超时模式下队空（且未关闭），take 失败。"""


class _End:
    """结束信号单例：take/take_all 在"队列空且已关闭"时返回它。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "END"

    def __bool__(self):
        return False


END = _End()


class BoundedMPMCQueue:
    """有界、阻塞、MPMC 安全的环形缓冲队列。"""

    def __init__(self, capacity, *, put_block=True, put_timeout=None):
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        if put_timeout is not None and put_timeout < 0:
            raise ValueError("put_timeout 不能为负")
        self._capacity = capacity
        self._buf = [None] * capacity  # 环形缓冲
        self._head = 0                 # 下一个出队位置
        self._count = 0                # 当前元素个数
        self._closed = False
        self._put_block = put_block
        self._put_timeout = put_timeout
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    # ------------------------------------------------------------------ #
    # 入队
    # ------------------------------------------------------------------ #
    def put(self, item, *, block=None, timeout=None):
        """放入一个任务。

        block/timeout 为 None 时取构造时的 put_block/put_timeout。
        队满时：block=True 阻塞等待空位（timeout 可限时，超时抛
        QueueFull）；block=False 立即抛 QueueFull。
        队列已关闭（包括等待期间被关闭）抛 QueueClosed。
        """
        if block is None:
            block = self._put_block
        if timeout is None:
            timeout = self._put_timeout

        with self._not_full:
            if self._closed:
                raise QueueClosed("队列已关闭，拒绝 put")
            if not block:
                if self._count == self._capacity:
                    raise QueueFull("队列已满")
            else:
                deadline = None if timeout is None else time.monotonic() + timeout
                while self._count == self._capacity:
                    if self._closed:
                        raise QueueClosed("等待空位期间队列被关闭")
                    if deadline is None:
                        self._not_full.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise QueueFull("put 超时，队列仍满")
                        self._not_full.wait(remaining)
                if self._closed:
                    raise QueueClosed("队列已关闭，拒绝 put")

            tail = (self._head + self._count) % self._capacity
            self._buf[tail] = item
            self._count += 1
            # 唤醒所有等待数据的消费者（见模块 docstring 的公平性说明）
            self._not_empty.notify_all()

    # ------------------------------------------------------------------ #
    # 出队
    # ------------------------------------------------------------------ #
    def take(self, *, block=True, timeout=None):
        """取走一个任务并返回。

        队列空且已关闭时返回结束信号 END。
        block=False 或超时后仍空（未关闭）抛 QueueEmpty。
        """
        with self._not_empty:
            if not block:
                if self._count == 0:
                    return END if self._closed else self._raise_empty()
            else:
                deadline = None if timeout is None else time.monotonic() + timeout
                while self._count == 0:
                    if self._closed:
                        return END
                    if deadline is None:
                        self._not_empty.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise QueueEmpty("take 超时，队列仍空")
                        self._not_empty.wait(remaining)
                if self._count == 0:  # 等待期间被关闭且队列已空
                    return END

            item = self._buf[self._head]
            self._buf[self._head] = None  # 释放引用，助 GC
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
            # 有空位了，唤醒所有等待的生产者
            self._not_full.notify_all()
            return item

    def take_all(self, *, block=True, timeout=None):
        """一次性取走当前队列中的全部任务，按入队顺序返回 list。

        队列空且已关闭时返回结束信号 END（不是空 list）。
        block=True 时若当前为空则阻塞，直到"至少有一个任务"或关闭，
        然后取走此刻的全部任务。
        """
        with self._not_empty:
            if not block:
                if self._count == 0:
                    return END if self._closed else self._raise_empty()
            else:
                deadline = None if timeout is None else time.monotonic() + timeout
                while self._count == 0:
                    if self._closed:
                        return END
                    if deadline is None:
                        self._not_empty.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise QueueEmpty("take_all 超时，队列仍空")
                        self._not_empty.wait(remaining)
                if self._count == 0:
                    return END

            items = []
            for _ in range(self._count):
                items.append(self._buf[self._head])
                self._buf[self._head] = None
                self._head = (self._head + 1) % self._capacity
            self._count = 0
            self._not_full.notify_all()
            return items

    # ------------------------------------------------------------------ #
    # 关闭与状态
    # ------------------------------------------------------------------ #
    def close(self):
        """优雅关闭：拒绝新 put，已入队任务仍可被消费。

        唤醒所有阻塞中的 put（抛 QueueClosed）和 take/take_all
        （队列空时返回 END）。幂等，可重复调用。
        """
        with self._lock:
            self._closed = True
            self._not_full.notify_all()
            self._not_empty.notify_all()

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def qsize(self):
        with self._lock:
            return self._count

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return self.qsize()

    @staticmethod
    def _raise_empty():
        raise QueueEmpty("队列为空")


# ====================================================================== #
# 多生产者多消费者示例：python3 fair_queue.py
# ====================================================================== #
if __name__ == "__main__":
    import random

    NUM_PRODUCERS = 4
    NUM_CONSUMERS = 3
    ITEMS_PER_PRODUCER = 250
    TOTAL = NUM_PRODUCERS * ITEMS_PER_PRODUCER

    q = BoundedMPMCQueue(capacity=16, put_block=True)  # 队满阻塞（背压）

    consumed = []          # 消费者取到的 (producer_id, seq)
    consumed_lock = threading.Lock()
    batch_takes = [0]      # take_all 命中次数（演示批量消费）

    def producer(pid):
        for seq in range(ITEMS_PER_PRODUCER):
            q.put((pid, seq))
            if seq % 50 == 0:
                time.sleep(random.random() * 0.001)  # 模拟不均匀生产速度
        print(f"[producer {pid}] 完成 {ITEMS_PER_PRODUCER} 个任务")

    def consumer(cid):
        # 消费者 0 用批量 take_all，其余用单个 take，两种路径都验证
        while True:
            if cid == 0:
                batch = q.take_all()
                if batch is END:
                    break
                batch_takes[0] += 1
                items = batch
            else:
                item = q.take()
                if item is END:
                    break
                items = [item]
            for it in items:
                time.sleep(random.random() * 0.0002)  # 模拟消费耗时
                with consumed_lock:
                    consumed.append(it)

    producers = [threading.Thread(target=producer, args=(i,)) for i in range(NUM_PRODUCERS)]
    consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(NUM_CONSUMERS)]

    for t in consumers + producers:
        t.start()
    for t in producers:
        t.join()

    q.close()  # 生产者全部完成 → 优雅关闭：余量消费完后消费者收到 END

    for t in consumers:
        t.join()

    # ---- 校验：每个任务恰好被消费一次 ----
    expected = {(pid, seq) for pid in range(NUM_PRODUCERS) for seq in range(ITEMS_PER_PRODUCER)}
    got = list(consumed)
    assert len(got) == TOTAL, f"数量不符: {len(got)} != {TOTAL}"
    assert len(set(got)) == TOTAL, "存在重复消费！"
    assert set(got) == expected, "存在丢失或多余的任务！"

    print(f"\nOK: {NUM_PRODUCERS} 生产者 x {ITEMS_PER_PRODUCER} = {TOTAL} 个任务，"
          f"{NUM_CONSUMERS} 个消费者，每个任务恰好消费一次。")
    print(f"批量消费次数（consumer 0 的 take_all 批数）: {batch_takes[0]}")

    # ---- 演示：关闭后的行为 ----
    try:
        q.put("late")
    except QueueClosed as e:
        print(f"关闭后 put 被拒: {e}")
    assert q.take() is END
    print("关闭且已空时 take 返回结束信号 END")

    # ---- 演示：非阻塞队满策略 ----
    q2 = BoundedMPMCQueue(capacity=2, put_block=False)
    q2.put(1)
    q2.put(2)
    try:
        q2.put(3)
    except QueueFull as e:
        print(f"非阻塞模式队满立即失败: {e}")
    assert q2.take_all() == [1, 2]
    print("take_all 按入队顺序批量取走: [1, 2]")
