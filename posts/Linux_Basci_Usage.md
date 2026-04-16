今天我们来介绍一些在实际开发工作中需要掌握的基础 Linux 开发环境，请注意我们接下来所介绍的内容基本满足企业“熟悉 Linux 开发环境”的要求，不涉及内核或运维深水区。我们按照日常使用频率进行。

1. 文件与目录

   这是非常基础的部分，我们需要熟练到不用查看文档：

   ```bash
   ls -l
   ls -lh
   tree
   
   cd /path
   cd ..
   cd ~
   
   pwd
   
   mkdir -p a/b/c
   rm -rf folder
   cp -r src dst
   mv old new
   ```

2. 查看文件

   ```bash
   cat file.txt
   less file.txt
   head -n 20 log.txt
   tail -n 50 log.txt
   tail -f train.log
   ```

   这部分最为常用的是：

   ```bash
   tail -f train.log
   ```

   实时查看训练日志。

3. 进程管理（必须掌握）

   查看进程：

   ```bash
   ps -ef
   ps -ef | grep python
   
   # or
   
   top
   htop
   ```

   杀死进程：

   ```bash
   kill PID
   kill -9 PID
   ```

4. GPU 管理

   必须熟练掌握：

   ```bash
   nvidia-smi
   
   # or
   
   gpustat
   ```

   持续查看：

   ```bash
   watch -n 1 nvidia-smi
   
   # or
   
   gpustat -i
   ```

   只看自己进程：

   ```bash
   nvidia-smi | grep python
   ```

   查看显存占用：

   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv
   ```

5. 后台训练

   正常我们不会在前台跑训练。

   ```bash
   nohup python train.py > train.log 2>&1 &
   ```

   含义：

   - nohup：后台运行
   - train.log：保存日志
   - 2>&1：保存错误
   - &：放入后台

   查看后台任务：

   ```bash
   jobs -l
   ```

6. tmux

   tmux 本质上是一个终端复用器（terminal multiplexer），它解决一个核心问题：SSH 断开后，程序仍然继续运行，并且你可以随时回来继续操作。在远程训练、长时间实验、服务器开发中，这是必备工具。

   没有 tmux 会发生什么？你在服务器运行训练：

   ```bash
   python train.py
   ```

   如果出现：

   - 断网
   - 关电脑
   - SSH 断开
   - VSCode 关闭

   训练会直接停止。这在开发中是不可接受的。

   如果使用 tmux：

   ```bash
   tmux new -s train
   python train.py
   ```

   即使：

   - 关闭电脑
   - 断开 SSH
   - 换网络

   训练仍然在服务器后台运行。你稍后可以回来：

   ```bash
   tmux attach -t train
   ```

   恢复到原来的终端界面。就像“远程屏幕保持器”。

   最常用 5 个命令：

   ```bash
   # 创建一个叫 train 的会话
   tmux new -s train
   
   # 分离 tmux（此时：你退出 tmux 但程序继续运行）
   # 按键：
   Ctrl + B
   D
   
   # 查看所有 session
   tmux ls
   
   # 恢复 session（回到训练界面）
   tmux attach -t train
   
   # 关闭 session
   exit
   
   # or
   
   Ctrl + D
   ```

   nohup 的优点是简单，但是它无法交互、无法回到终端且不方便调试。tmux 解决了上述问题，可恢复终端、可交互、可调试、可多窗口。

7. 远程服务器开发

   登陆服务器：

   ```bash
   ssh user@ip
   
   # 指定端口
   
   ssh user@ip -p 22
   ```

   上传文件：

   ```bash
   scp file.py user@ip:/path/
   
   # 上传目录
   scp -r project user@ip:/path/
   ```

   下载文件：

   ```bash
   scp user@ip:/path/file .
   ```

   更快同步：

   ```bash
   rsync -avz project user@ip:/path
   
   # 这条命令用于把本地 project 目录同步到远程服务器，而且是增量同步（只传变化的文件）。
   # 它比 scp 高效得多
   
   # 完整例子
   
   # 本地：/home/user/project
   
   # 远程：/home/lab/
   
   # 执行：rsync -avz project lab@192.168.1.10:/home/lab/
   
   # 结果：远程变成：/home/lab/project
   ```

8. 环境管理

   无论是高校还是企业，用的最多的是 conda。

   ```bash
   # 创建环境
   conda create -n dl python=3.10
   
   # 激活
   conda activate dl
   
   # 退出
   conda deactivate
   
   # 删除
   conda remove -n dl --all
   ```

9. 查日志

   grep 是必须掌握的。

   ```bash
   grep "loss" train.log
   
   # 显示行号
   grep -n "loss" train.log
   
   # 实时查
   tail -f train.log | grep loss
   
   # 递归查代码
   grep -r "forward" .
   ```

10. 磁盘与资源查看

    查看磁盘：

    ```bash
    df -h
    ```

    查看目录大小：

    ```bash
    du -h --max-depth=1
    ```

    排序：

    ```bash
    du -h | sort -h
    ```

    查看内存：

    ```bash
    free -h
    ```

11. 权限管理（必须会）

    增加执行权限：

    ```bash
    chmod +x train.sh
    ```

    所有权限：

    ```bash
    chmod 777 file
    ```

    修改所有者：

    ```bash
    chown user file
    ```

12. bash脚本（基础即可）

    例如 train.sh ：

    ```bash
    #!/bin/bash
    
    python train.py \
        --lr 1e-4 \
        --batch_size 32 \
        --epoch 100
    ```

    运行：

    ```bash
    bash train.sh
    ```

13. 压缩与解压

    ```bash
    # 解压 tar
    tar -xvf file.tar
    
    # 解压 gz
    tar -xzvf file.tar.gz
    
    # 压缩
    tar -czvf data.tar.gz data/
    
    # 解压 zip
    unzip file.zip
    ```

14. 软链接（非常常用）

    ```bash
    ln -s /data/dataset dataset
    ```

    很多代码依赖这个。软链接（symbolic link / symlink）本质是一个指向另一个文件或目录的快捷引用。它类似 Windows 的“快捷方式”，但更底层、更强大，在 Linux 项目中非常常用。

    创建命令：

    ```bash
    ln -s 源路径 软链接名
    ```

    我们举一个最简单的例子：

    ```bash
    # 假设真实数据在：/data/datasets/ImageNet
    
    # 但你的项目在：/home/user/project
    
    # 你不想复制 1TB 数据，就创建软链接：
    ln -s /data/datasets/ImageNet ./dataset
    
    # 结果：
    project/
     ├── train.py
     └── dataset -> /data/datasets/ImageNet
    
    # 注意：dataset 只是一个“指针”，不占空间。
    # 但代码可以直接：path = "dataset/train"
    # 就像真的在项目里一样。
    
    # 一个非常重要的坑（相对路径）
    # 推荐使用绝对路径：ln -s /data/dataset dataset
    # 不推荐：ln -s ../dataset dataset
    # 因为：移动目录会失效。
    ```

    

15. 环境变量（了解即可）

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    
    # 多卡
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    # 临时运行
    CUDA_VISIBLE_DEVICES=0 python train.py
    ```

以上基本上就是我们在开发过程中使用频率最高的一些命令了。

------

下面我们再介绍一个补充内容：Vim。

在远程服务器上无法使用 GUI 编辑器时，最常用的就是 Vim。例如修改配置文件、快速改代码、修复 bug 等。

打开文件：

```bash
vim train.py
```

如果文件不存在，会新建。

Vim 有三个核心模式：

- 普通模式（Normal）  ← 默认进入
- 插入模式（Insert）  ← 编辑文本
- 命令模式（Command） ← 保存退出

必须理解这三种模式，否则会感觉 Vim 很难用。

------

一、进入插入模式（开始编辑）

在普通模式下按（注意以下所有命令请将键盘输入切换为英文）：

```
i
```

进入插入模式，就可以像普通编辑器一样打字。

------

二、退出插入模式

按：

```
Esc
```

回到普通模式。这是最重要的一步。

------

三、保存退出

在普通模式下输入：

```
: wq
```

然后回车。

------

四、常用命令（必须掌握）

保存

```
:w
```

退出

```
:q
```

强制退出（不保存）

```
:q!
```

保存并退出

```
:wq
```

------

五、最小使用流程

编辑文件：

```
vim train.py
```

步骤：

```
i        # 进入编辑
修改代码

Esc      # 退出编辑

:wq      # 保存退出
```

这就是完整使用流程。

------

六、常用移动操作

上下左右：

```
h  左
j  下
k  上
l  右
```

更快的：

跳到行首

```
0
```

跳到行尾

```
$
```

跳到文件开头

```
gg
```

跳到文件末尾

```
G
```

------

七、删除操作

删除一行：

```
dd
```

删除 5 行：

```
5dd
```

删除一个单词：

```
dw
```

------

八、复制粘贴

复制一行：

```
yy
```

粘贴：

```
p
```

复制 5 行：

```
5yy
```

------

九、搜索

搜索字符串：

```
/loss
```

按：

```
n
```

跳到下一个匹配。