class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next is not None:
            current = current.next
        current.next = new_node

def list_to_linked_list(input_list):
    linked_list = LinkedList()
    for item in input_list:
        linked_list.append(item)
    return linked_list

original_list = [1, 2, 3, 4, 5]

# Convert the list to a linked list
head_of_linked_list = list_to_linked_list(original_list)
p = head_of_linked_list.head
print(p.data)
p = p.next
print(p.data)
p = head_of_linked_list.head
print(p.data)