class Node(object):
    """ 节点 """

    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkedlist(object):
    """ 单链表 """

    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        """ 判断链表是否为空 """
        return self.__head == None

    def length(self):
        """ 获取链表长度 """
        # cur指针，移动来遍历元素
        cur = self.__head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """ 遍历链表元素 """
        cur = self.__head
        while cur != None:
            print(cur.elem, end=" ")
            cur = cur.next
        print(" ")

    def add(self, item):
        """ 头插法 """
        node = Node(item)
        # node先指向 原头部节点
        node.next = self.__head
        # 再让 头指针指向新插入的node节点，这样node成为了新的头部节点
        self.__head = node

    def append(self, item):
        """ 尾插法， 链表尾部添加元素 """
        # item仅为数据
        node = Node(item)
        if self.is_empty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        """ 指定位置插入 """
        # <0 头插法
        if pos <= 0:
            self.add(item)
        # >length， 尾插法
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            pre = self.__head
            count = 0
            # 将pre指针停在pos-1位置，
            while count < (pos - 1):
                count += 1
                pre = pre.next
            node = Node(item)
            node.next = pre.next
            pre.next = node

    def remove(self, item):
        cur = self.__head
        pre = None
        while cur != None:
            if cur.elem == item:
                # 先判断删除的节点是否为头结点
                if cur == self.__head:
                    self.__head = cur.next
                else:
                    # 删除的点若不是头结点
                    pre.next = cur.next
                break
            else:
                # 移动两个指针
                pre = cur
                cur = cur.next

    def search(self, item):
        cur = self.__head
        while cur != None:
            if cur.elem == item:
                print("num is existed")
            else:
                cur = cur.next
        print("Not contain this num")


if __name__ == "__main__":
    sll = SingleLinkedlist()
    print(sll.is_empty())
    print(sll.length())

    sll.append(11)
    print(sll.length())

    sll.append(12)
    sll.append(13)
    sll.append(14)
    sll.append(15)
    # sll.travel()

    sll.add(321)
    sll.travel()

    sll.insert(-1, 11)
    sll.insert(2, 222)
    sll.insert(23, 999)
    sll.travel()

    sll.remove(321)
    sll.travel()

    sll.search(9999)
