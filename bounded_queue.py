"""bounded_queue.py —— 多生产者/多消费者有界阻塞队列（仅标准库，纯自实现）。

设计要点
========
存储
    定长 list 实现的环形缓冲（head 指针 + size 计数），容量固定，
    不使用 queue.Queue，也不使用 collections.deque。

接口
    put(item, timeout=None)  入队；关闭后抛 QueueClosed；
                             超时/立即失败策略下失败抛 QueueFull。
    take(timeout=None)       出队一个；队列空且已关闭时返回结束信号 CLOSED；
                             传 timeout 时超时抛 TimeoutError。
    take_all()               原子地一次性取走当前缓冲里的全部任务，返回 list
                             （可能为空 list；关闭后同样返回剩余任务）。
    close()                  优雅关闭：已入队任务照常消费完，
                             等待中的生产者收到 QueueClosed，
                             缓冲排空后等待中的消费者收到 CLOSED。

公平性（为什么不会饿死）
    threading.Condition.notify() 并不保证按等待先后唤醒，直接用
    「一把锁 + 两个 Condition + notify/notify_all」在理论上可能让某个
    线程反复被插队（饿死）。这里的做法：
      1. 每个等待者持有自己的 Condition，挂进显式的 FIFO 等待队列
         （生产者一条、消费者一条）；
      2. 只唤醒队首等待者（notify_head），被唤醒者完成操作后若条件仍
         成立则级联唤醒下一个队首；
      3. 新到的线程只要等待队列非空就必须排队，禁止插队（barging）；
      4. 超时/关闭退出的等待者置 cancelled 标记，peek 时惰性跳过。
    因此服务顺序严格 FIFO：先等的生产者先拿到空位、先等的消费者先
    拿到任务，不存在被无限插队的情形。
    例外：take_all 是刻意的「批量抢占」操作，语义就是取走当前全部，
    不参与逐个消费的公平排队（调用方应清楚这一点）。

队满时 put 的策略（构造时 put_policy 选择）
    "block"   无限阻塞直到有空位或队列关闭。最省心、不丢任务，
              天然形成背压；代价是消费者停摆时生产者会被一直挂住。
    "timeout" 阻塞至多 put_timeout 秒，超时抛 QueueFull。等待有上界，
              生产者可自行重试或降级；代价是要调参，瞬时高峰可能误失败。
    "fail"    队满立即抛 QueueFull。生产者延迟最低；代价是调用方必须
              自己处理丢弃/重试（重试循环本质上就是自己实现阻塞）。
    单次 put 也可用 timeout= 参数覆盖构造时的策略。

运行示例：python3 bounded_queue.py
"""

from __future__ import annotations

import threading
import time

__all__ = ["BoundedQueue", "QueueClosed", "QueueFull", "CLOSED"]


class QueueClosed(Exception):
    """队列已关闭，拒绝新的 put。"""


class QueueFull(Exception):
    """put 在 timeout/fail 策略下未能入队。"""


class _ClosedSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "CLOSED"


CLOSED = _ClosedSentinel()
"""take 的结束信号：队列已关闭且缓冲已排空时返回此单例。"""


class _Waiter:
    """一个等待者：私有 Condition + 取消标记。"""

    __slots__ = ("cond", "cancelled")

    def __init__(self, lock: threading.Lock) -> None:
        self.cond = threading.Condition(lock)
        self.cancelled = False


class _WaitQueue:
    """等待者的 FIFO（定长数组 + 头指针，惰性清理，不用 deque）。"""

    __slots__ = ("_items", "_head")

    def __init__(self) -> None:
        self._items: list[_Waiter] = []
        self._head = 0

    def push(self, waiter: _Waiter) -> None:
        self._items.append(waiter)

    def peek(self) -> _Waiter | None:
        items = self._items
        head = self._head
        while head < len(items) and items[head].cancelled:
            head += 1
        self._head = head
        if head >= 16 and head * 2 >= len(items):  # 惰性压缩，防止数组无限增长
            del items[:head]
            self._head = 0
        return items[self._head] if self._head < len(items) else None

    def remove(self, waiter: _Waiter) -> None:
        waiter.cancelled = True  # 惰性删除，由 peek 跳过

    def notify_head(self) -> None:
        waiter = self.peek()
        if waiter is not None:
            waiter.cond.notify()

    def notify_all_and_clear(self) -> None:
        for waiter in self._items[self._head:]:
            if not waiter.cancelled:
                waiter.cancelled = True
                waiter.cond.notify()
        self._items = []
        self._head = 0


class BoundedQueue:
    """容量固定的 MPMC 阻塞队列，支持批量取与优雅关闭。"""

    def __init__(
        self,
        capacity: int,
        *,
        put_policy: str = "block",
        put_timeout: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        if put_policy not in ("block", "timeout", "fail"):
            raise ValueError("put_policy 必须是 'block' / 'timeout' / 'fail'")
        if put_policy == "timeout" and put_timeout is None:
            raise ValueError("put_policy='timeout' 时必须给出 put_timeout")
        self._capacity = capacity
        self._buf: list = [None] * capacity  # 环形缓冲
        self._head = 0                       # 队首下标
        self._size = 0                       # 当前元素个数
        self._lock = threading.Lock()
        self._not_empty = _WaitQueue()       # 消费者 FIFO
        self._not_full = _WaitQueue()        # 生产者 FIFO
        self._closed = False
        self._put_policy = put_policy
        self._put_timeout = put_timeout

    # ---------------------------------------------------------------- 状态

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def qsize(self) -> int:
        with self._lock:
            return self._size

    def __len__(self) -> int:
        return self.qsize()

    @staticmethod
    def _is_turn(waitq: _WaitQueue, waiter: _Waiter | None) -> bool:
        """是否轮到当前线程：无等待者时可直接进入；否则必须是队首。

        被 close 强制唤醒（cancelled）的等待者允许直接竞争——此时只需
        保证每个任务被消费恰好一次（由锁保证），不再要求 FIFO 次序。
        """
        if waiter is None:
            return waitq.peek() is None
        if waiter.cancelled:
            return True
        return waitq.peek() is waiter

    # ---------------------------------------------------------------- put

    def put(self, item, timeout: float | None = None) -> None:
        """入队一个任务。

        关闭后抛 QueueClosed；timeout/fail 策略下等不到空位抛 QueueFull。
        """
        if item is CLOSED:
            raise ValueError("CLOSED 是保留的结束信号，不能作为任务入队")
        if timeout is None:
            if self._put_policy == "fail":
                timeout = 0.0
            elif self._put_policy == "timeout":
                timeout = self._put_timeout

        with self._lock:
            waiter: _Waiter | None = None
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if self._closed:
                    if waiter is not None:
                        self._not_full.remove(waiter)
                    raise QueueClosed("队列已关闭，put 失败")
                if self._size < self._capacity and self._is_turn(self._not_full, waiter):
                    break
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if waiter is not None:
                            self._not_full.remove(waiter)
                        raise QueueFull("队满，put 超时/立即失败")
                    if waiter is None:
                        waiter = _Waiter(self._lock)
                        self._not_full.push(waiter)
                    if not waiter.cond.wait(remaining):
                        self._not_full.remove(waiter)
                        if self._size < self._capacity:  # 弥补被超时吞掉的唤醒
                            self._not_full.notify_head()
                        raise QueueFull("队满，put 超时/立即失败")
                else:
                    if waiter is None:
                        waiter = _Waiter(self._lock)
                        self._not_full.push(waiter)
                    waiter.cond.wait()

            if waiter is not None:
                self._not_full.remove(waiter)
            tail = (self._head + self._size) % self._capacity
            self._buf[tail] = item
            self._size += 1
            if self._size < self._capacity:  # 级联：还有空位就唤醒下一位生产者
                self._not_full.notify_head()
            self._not_empty.notify_head()    # 有货了，唤醒队首消费者

    # ---------------------------------------------------------------- take

    def take(self, timeout: float | None = None):
        """取走一个任务；队列空且已关闭时返回 CLOSED。

        传入 timeout 时，超时仍未取到抛 TimeoutError。
        """
        with self._lock:
            waiter: _Waiter | None = None
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if self._size > 0 and self._is_turn(self._not_empty, waiter):
                    break
                if self._closed and self._size == 0:
                    if waiter is not None:
                        self._not_empty.remove(waiter)
                    return CLOSED
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if waiter is not None:
                            self._not_empty.remove(waiter)
                        raise TimeoutError("take 超时")
                    if waiter is None:
                        waiter = _Waiter(self._lock)
                        self._not_empty.push(waiter)
                    if not waiter.cond.wait(remaining):
                        self._not_empty.remove(waiter)
                        if self._size > 0:  # 弥补被超时吞掉的唤醒
                            self._not_empty.notify_head()
                        raise TimeoutError("take 超时")
                else:
                    if waiter is None:
                        waiter = _Waiter(self._lock)
                        self._not_empty.push(waiter)
                    waiter.cond.wait()

            if waiter is not None:
                self._not_empty.remove(waiter)
            item = self._buf[self._head]
            self._buf[self._head] = None  # 释放引用，帮助 GC
            self._head = (self._head + 1) % self._capacity
            self._size -= 1
            if self._size > 0:  # 级联：还有货就唤醒下一位消费者
                self._not_empty.notify_head()
            if not self._closed:  # 腾出空位，唤醒队首生产者
                self._not_full.notify_head()
            return item

    def take_all(self) -> list:
        """原子地一次性取走当前缓冲中的全部任务，按入队顺序返回 list。

        非阻塞：当前为空就返回 []。调用方可结合 closed 判断是否结束。
        注意这是刻意的批量抢占操作，不参与逐个消费的 FIFO 排队。
        """
        with self._lock:
            n = self._size
            items = [self._buf[(self._head + i) % self._capacity] for i in range(n)]
            for i in range(n):
                self._buf[(self._head + i) % self._capacity] = None
            self._head = 0
            self._size = 0
            if not self._closed:
                self._not_full.notify_head()  # 腾出大量空位，由 put 内级联扩散
            return items

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        """优雅关闭：已入队任务仍可消费，新 put 抛 QueueClosed。

        等待中的生产者立即收到 QueueClosed；缓冲排空后，
        等待中的消费者从 take 收到 CLOSED。
        """
        with self._lock:
            self._closed = True
            self._not_empty.notify_all_and_clear()
            self._not_full.notify_all_and_clear()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"BoundedQueue(size={self._size}, capacity={self._capacity}, "
                f"closed={self._closed})"
            )


# ==================================================================== 示例

def _demo_mpmc() -> None:
    """多生产者多消费者：验证每个任务恰好被消费一次。"""
    num_producers = 4
    items_per_producer = 250
    total = num_producers * items_per_producer

    q = BoundedQueue(capacity=8, put_policy="block")
    consumed: list = []
    consumed_lock = threading.Lock()

    def producer(pid: int) -> None:
        for seq in range(items_per_producer):
            q.put((pid, seq))

    def consumer() -> None:
        while True:
            item = q.take()
            if item is CLOSED:  # 结束信号：队列已关闭且排空
                return
            with consumed_lock:
                consumed.append(item)

    def batch_consumer() -> None:
        while True:
            batch = q.take_all()  # 一次取走当前全部
            if batch:
                with consumed_lock:
                    consumed.extend(batch)
            elif q.closed:
                return
            else:
                time.sleep(0.001)

    producers = [threading.Thread(target=producer, args=(i,)) for i in range(num_producers)]
    consumers = [threading.Thread(target=consumer) for _ in range(2)]
    consumers.append(threading.Thread(target=batch_consumer))

    for t in producers + consumers:
        t.start()
    for t in producers:
        t.join()
    q.close()  # 生产者全部完成后优雅关闭
    for t in consumers:
        t.join()

    expected = {(p, s) for p in range(num_producers) for s in range(items_per_producer)}
    assert len(consumed) == total, f"数量不符: {len(consumed)} != {total}"
    assert set(consumed) == expected, "存在丢失或重复消费的任务"
    print(f"[demo1] OK: {num_producers} 生产者 x {items_per_producer} 任务，"
          f"3 个消费者（含 1 个批量消费者），{total} 个任务每个恰好消费一次。")


def _demo_put_policies() -> None:
    """演示队满时 timeout / fail 两种非阻塞策略。"""
    q = BoundedQueue(capacity=2, put_policy="timeout", put_timeout=0.05)
    q.put("a")
    q.put("b")
    try:
        q.put("c")  # 队满，等 50ms 后失败
    except QueueFull as exc:
        print(f"[demo2] timeout 策略: put 被拒绝 -> {exc}")

    q2 = BoundedQueue(capacity=1, put_policy="fail")
    q2.put("x")
    try:
        q2.put("y")  # 队满，立即失败
    except QueueFull as exc:
        print(f"[demo2] fail 策略:    put 被拒绝 -> {exc}")

    print(f"[demo2] 关闭前: {q!r}")
    q.close()
    try:
        q.put("z")
    except QueueClosed as exc:
        print(f"[demo2] 关闭后 put:   -> {exc}")
    assert q.take() == "a"
    assert q.take() == "b"
    assert q.take() is CLOSED  # 已入队任务消费完后返回结束信号
    print("[demo2] 关闭后已入队任务照常消费，排空后 take 返回 CLOSED。")


if __name__ == "__main__":
    _demo_mpmc()
    _demo_put_policies()
