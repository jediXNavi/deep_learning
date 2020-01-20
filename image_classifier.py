def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

######################################################
####### Do not modify the code above this line #######
######################################################

class cifar_resnet20(nn.Module):
    def __init__(self):
        super(cifar_resnet20, self).__init__()
        ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
        url = 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth'
        ResNet20.load_state_dict(model_zoo.load_url(url))
        modules = list(ResNet20.children())[:-1]
        backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.shape[0], -1)
        return self.fc(out)

def train_img_classifier():
      trainset = datasets.CIFAR10('./data', download=True, transform=transform)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=400,
                                          shuffle=True, num_workers=2)
      ## Do the training
      for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
          running_loss = 0.0
          for i, data in enumerate(trainloader, 0):
              # get the inputs
              inputs, labels = data

              # zero the parameter gradients
              optimizer.zero_grad()

              # forward + backward + optimize
              outputs = model(inputs.to(device))
              loss = criterion(outputs.to(device), labels.to(device))
              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              if i % 20 == 19:    # print every 20 mini-batches
                  print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                  running_loss = 0.0
      print('Finished Training')

def testing_img_classifier():

  import torch
  import torchvision
  import io
  import base64
  import numpy as np
  from torch.nn import functional as fn
  from torchvision import transforms as tr

  def convert_to_pil(image):
      pil_image = tr.ToPILImage()(image.squeeze())
      return pil_image

  def imshow(img):
      img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()

  def make_png_b64(image):
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format = "PNG")
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    base64_encoded_result_bytes = base64.b64encode(img_bytes)
    base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
    return base64_encoded_result_str

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = cifar_resnet20().to(device)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                           shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  PATH = './cifar_net.pth'
  #Creating the HTML
  html_table = """<html><table style=width:100%><tr><th width=100>Image</th><th>Actual</th><th>Predicted</th>
                   <th>plane</th><th>car</th><th>bird</th><th>cat</th>
                   <th>deer</th><th>dog</th><th>frog</th><th>horse</th><th>ship</th><th>truck</th></tr>
                 """

  test_model = model.load_state_dict(torch.load(PATH))
  model.eval()

  total = 0
  correct = 0
  print("SUcess 4")
  output_data = ''
  for i, data in enumerate(testloader, 0):
    count = 0
    images, labels = data
    labels = labels.to(device)
    images = images.to(device)
    #   imshow(torchvision.utils.make_grid(images))
    #   # print labels
    #   print(' '.join('%5s' % classes[labels[j]] for j in range(1)))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.to(device).size(0)
    correct += (predicted == labels.to(device)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    with torch.no_grad():
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      actual_class = classes[labels]
      print(actual_class)
      prob = fn.softmax(outputs, dim=-1)
      print(prob)
      prob_array = prob.to("cpu").numpy()
      print(prob_array)
      _, predicted = torch.max(outputs, 1)
      predicted_class = classes[predicted]
      print(predicted_class)
      p_image = convert_to_pil(images.to("cpu"))
      print(type(p_image))
      b_string = make_png_b64(p_image)
      print(b_string)

      output_data = output_data + """<tr><td><img width=100 src=data:image/png;base64,"""+b_string+"""></td><td align='center'>"""+actual_class+"""</td><td align='center'>"""+predicted_class+"""</td>"""
      for pb in np.nditer(prob_array):
        output_data = output_data + """<td align='center'>"""+str(np.round(pb,3))+"""</td>"""
      output_data = output_data + "</tr>"


    if i%20==19:
      break
  html_table = html_table + output_data
  html_table = html_table + "</tr></table></html>"


  hs = open("ClassificationHTML.html",'w+')
  hs.write(html_table)

if __name__ == '__main__':

    model = cifar_resnet20().to(device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9)
    train_model = sys.argv[1]
    train_model == 'False'
    if train_model == 'True':
      train_img_classifier()
    else:
      testing_img_classifier()
