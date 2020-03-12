class Node(object):
    def __init__(self, item):
        self.elem = item
        self.prev = None
        self.next = None


class DoubleLinkedList(object):
    def __int__(self):
        self.__head = node

    def is_empty(self):
        return self.__head is None

    def length(self):
        cur = self.__head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        cur = self.__head
        while cur != None:
            print(cur.elem, end=" ")
            cur =cur.next

    def add(self, item):
        """ 头插法 """
        node = Node(item)
        # 新节点指向原头结点
        node.next = self.__head
        # 头指针指向新节点
        self.__head = node
        # 原头节点的prev指向新节点
        node.next.prev = node

    def append(self, item):
        """ 尾插法 """
        node = Node(item)
        if self.is_empty():
            self.__head = node
        else:
            cur =self.__head
            while cur != None:
                cur = cur.next
            cur.next = node
            node.prev = cur

