# 函数具体计算方案
## analyse_resources_first_time
1. 首先从8(一块bram能够支持矩阵边长)开始，每次增加都是乘二，遍历最大支持的矩阵边长，直到512
2. 按照A:B:C=1:1:4计算各个矩阵占用空间。并计算各个矩阵占用的bram。其中A和B占用bram数为矩阵边长除以8。
    C占用bram数为C占用空间除以每块bram容量(此时不考虑C的带宽问题)
3. 计算ABC占用的bram总数，如果bram足够，则持续增加矩阵边长，直到bram不够。得到芯片能够支持的最大矩阵边长
4. 如果矩阵边长达到了512，则构成一个完整的bram组。此后不再依据矩阵边长来计算，而是以bram组为单位进行扩展。
    仍旧是每次增加，bram组数乘二，直到找到支持的最大的bram组数。在此期间，保持矩阵边长不变
5. 分配完成后，对bram组进行4合1操作。每次合并，bram组数除以4，最大矩阵边长乘2
6. 根据最大矩阵边长计算最小矩阵边长，使得最小矩阵也能填满bram组的一行
7. 检查计算资源是否足够。按照mult占用61个lut，add 8个，sub 8个，每个dsp抵消25个计算。将各个计算单元
    占用的lut相加得到总lut，再将去dsp抵消的，与可用lut比较，检测是否足够
8. 如果不够，缩减bram，减小矩阵边长。
    - 如果bram_group为1，且边长大于512，则将bram组拆分为4组，然后减半。最大矩阵边长减半
    - 如果bram_group为2，则bram组减半
    - 如果bram_group为1，且边长为512，则bram组减为0。最大矩阵边长减半
    - 如果边长小于512，则边长减半
    一直减少知道lut足够
   如果够，尝试给每组bram的计算单元翻倍，使每组拥有超过1组计算单元
   最终找到合适的bram组数，矩阵边长，和每组bram分配的计算单元组数
   重新计算bram占用
9. 此时初步分析完成。得到以下结果：
    - 完整的bram组数(即边长为最大矩阵边长的组数)
    - 不完整bram组的长度(与上一条这两条之间，只有一个有意义)
    - 需要的总bram数(此时没有考虑C的带宽问题)
    - 可用bram总数
    - 最大支持的矩阵边长
    - 最小支持矩阵边长(为保证单个矩阵能够填满bram的一行，该值需不小于最大支持的矩阵边长的平方根)
    - 每个bram组的计算单元组数
    - 需要的总lut数(仅使用lut计算时的lut占用，不考虑dsp抵消的)
    - 可用的lut数(因为上一条展示的时候不考虑dsp抵消的，而计算的时候考虑，所以可能出现上一条比这条大的情况)
10. 因为之前计算C的bram占用的时候没有考虑C的带宽带来的bram空间浪费的问题，而是仅根据容量来计算，所以这里需要对C的bram占用进行修正
11. 按照以下三个原则对矩阵进行切块
    - 尽量切大块，但不超过片上支持的最大边长
    - 最小边长不能小于片上支持的最小边长
    - 矩阵的相乘边要为2的幂
    - 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
12. 对于切块原则进行详细描述
    - 初始时将左上角设为起始点
    - 从起始点开始分别在横向上和纵向上计算当前应该切的长度。横向和纵向的计算方式相同
    - 如果当前方向上的剩余长度大于等于片上支持的最大矩阵长度，则按照最大矩阵长度切分
    - 否则，如果小于最小矩阵长度，则按照最小矩阵长度切分(这会导致实际切下的长度大于矩阵长度，多切的部分补0)
    - 否则，如果是2的幂，则按照当前长度切分
    - 否则，找到刚好比当前长度大的2的幂L和刚好比当前长度小的2的幂S。如果当前长度大于(L+S)/2，则按L切分，否则按S切分
    - 切下一块后向右移动起始点。如果到达尽头，则将起始点的横向坐标重置为0，并向下移动起始点
    AB均切完后得到切块结果
13. 对切块的结果进行校验，保证A和B的相乘边(A的上边和B的左边)切分结果是一样的
14. 重新计算C需要的空间
    - 1. 从A和B的相乘边中找到边长最短的矩阵块
    - 根据bram块带宽和矩阵块最小边长计算bram块每周期输出的最大矩阵行数，即每周期每个bram组每组计算单元输出的结果数
    - 2. 在上面结果基础上乘每组计算单元组数，得到每个bram组每周期输出的结果数
    - 3. 根据C需要的带宽计算C实际需要的bram数
    - 结果为32bit的条件下，每列bram每周期能写入的结果数设为2
    - 用每bram组每周期输出的结果数，除以每周期能写入的结果数，得到C需要每个bram组的列数
    - 根据最大矩阵边长的平方乘4，得到每个bram组中C需要的空间
    - 用每bram组中C需要的空间，除以每bram组中C需要的列数，得到每bram组中C需要每列的空间
    - 再除以每列的宽度，得到每列的深度
    - 查表得到该深度下需要的bram36和bram18的个数
    - 再将这个个数乘以每组列数，得到每组C需要的bram数
    - 再计算每组AB需要的bram数：如果没有完整组，则为边长除以8。否则为边长除以8再乘深度方向的个数(
        比如1024，横向需要128，纵向需要2，共需要256)
    - 加和得到每组需要的bram总数
    - 如果bram组数大于等于1，再乘上组数，得到需要的bram总数
    此时得到修正后的bram总数
15. 如果够用，结束，返回结果
16. 如果不够用，则bram减半，重新分配lut，然后回到10重新修正











<!-- # rtl_gen
## 流程
1. 读取参数
2. 读取计算图和权重
3. 根据计算图推算im2col后矩阵尺寸
4. 根据矩阵尺寸及片上资源分析bram和lut的使用
5. 根据资源分配结果切分矩阵
6. 生成张量表达式，详细描述计算流程
7. 生成代码 -->




<!-- ## 关于张量表达式的生成和资源自动分配

### 1. 计算各层矩阵大小
根据计算图推算各层卷积需要计算的最大分块矩阵边长
对于某一层卷积，其需要计算的最大分块矩阵边长由im2col后两个矩阵的边长最小值决定
再在各层的最小值中选出最大的

### 2. 第一次资源分配
1. 按照A:B:C=1:1:4分配bram，其中A和B占用的bram块数由带宽决定。C由于不需要大带宽，可以根据容量决定
2. 除非计算资源不够，否则尽可能增加A和B的带宽。
3. 由于bram深度为512，所以如果A和B的深度超过了512，就浪费了带宽，所以限制A和B的最大深度为512，即单组
bram最大支持512x512的矩阵乘法。多余的bram另外成组
4. 如果bram资源足够容纳4组512，则可以进行最大1024的矩阵乘法。除非计算资源不够。更大的以此类推
5. 但很多网络其实用不到这么大的矩阵乘法，那么C其实用不到4份bram。可以根据实际占用缩减C的空间，分给AB。
注意：此时C占用的bram数仅根据容量决定，没有考虑带宽问题，可能小于实际需要的bram数
#### 修正C占用的bram数
##### 拆分张量表达式
对矩阵进行切块
1. 尽量切大块，但不超过片上支持的最大边长
2. 矩阵的相乘边要为2的幂
3. 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
返回切块结果
##### 资源分配
根据切块结果得到需要进行的最小矩阵乘法边长。该边长决定了C需要的带宽
根据C需要的带宽计算C实际需要的bram数
如果片上资源能够容纳修正后的ABC，则完成
如果不能容纳，则减少ABC的bram分配，然后回到上一步重新修正

### 3. 第一次拆分张量表达式
对矩阵进行切块
1. 尽量切大块，但不超过片上支持的最大边长
2. 矩阵的相乘边要为2的幂
3. 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
切分之后，如果切出来的矩阵边长均小于片上支持的最大矩阵边长，则考虑缩减C的空间
缩减方式：找到一个计算流程，使得
1. 尽可能多累加。
2. 传输时尽可能填满A和B
根据该计算流程计算C的峰值占用空间

### 4. 第二次资源分配
根据上一步计算出来的C的峰值占用，将C始终空闲的部分拿出去给其他部分使用：
1. 如果AB带宽未达到512，则考虑加给AB，但仍需保证AB的带宽为2的幂
2. 如果已经达到或者无法保证AB的带宽为2的幂，则单独成块，需保证新块的带宽为2的幂

如果发现C再多分出去一点，就能使新块的带宽再上一级，或是能使旧块的带宽再上一级，
则计算一下这种更激进的情况下的C的数值，然后和张量表达式生成器商量
否则资源分配到此结束

### 5. 第二次拆分张量表达式
在资源的二次分配给出的条件下，寻找新的计算方案
并评估新方案与旧方案相比的损失
根据每一次传输、内存拷贝、片上计算、启动DMA所需的总时间来估计
然后给出结论，是否同意C再多分出去一点

### 6. 第三次资源分配
根据张量表达式二次分配的结果，进行最终分配

### 7. 第三次拆分张量表达式
根据资源分配最终结果，切分张量表达式 -->








<!-- ## 关于张量表达式的生成和资源自动分配

### 首先根据计算图计算各层卷积大小
根据计算图中的数据，计算使用im2col后，各层卷积转换的矩阵大小
由于矩阵边长的限制，分块后单个矩阵的最大边长是有限的，
比如resnet18中，所需要计算的最大矩阵边长是512，
那么当bram足够时，创建一块支持1024x1024的bram，就不如创建
4块支持512x512的bram，因为后者可以提供更大的带宽
但其实如果计算图中有1024边长的矩阵，似乎也是创建4个512x512更快
但考虑到切块越小，总传输量越大的问题，所以不能按普通的512x512来处理
所谓切块越小总传输量越大，是因为切成小块之后，如果每一块传到片上之后只使用一次，
那就需要重复传输这一块。而切成大块之后，传上去之后相当于里面的每个小块都使用了很多次
所以将1024x1024切成512x512之后，为了不增加总传输量，就要求每一块传上去之后
需要重复使用，这就要求即便是切成512x512，仍要按照原来的1024x1024的数据去传，
即每次传输需要把原1024x1024切成的4个512x512方块传上去，而不能按行优先
传一行的4个512x512方块。这样子相当于还是计算1024x1024，同时能够获得更大的带宽
但这会带来一个问题，对于从1024x1024切出来的4块矩阵，需要计算A00xB00+A01xB10
对于仅有512的芯片来说，按顺序计算然后累加就可以。但在能够并行4个512x512的芯片上，
是可以同时计算A00xB00和A01xB01的，这就要求计算出来之后直接相加，所以实际上还是需要
1024的乘累加树。
所以综上所述，为了带宽，切块最大512。但乘累加树大小是和最大矩阵边长一样的。

### 在知道资源数量后，首先进行一次简单的分配
此时假设只进行方阵计算。目前保证AB的带宽为2的幂
对于矩阵A，设希望提供的带宽为$8n$，则：
当$n<=64$时，A和B各需要n块BRAM
从1开始遍历n
A，B各占用n块bram。A的数据量为$(8n)^2=64n^2Bytes$，则C的数据量为A的4倍，即$256n^2Bytes$。
需要的bram数量为$256n^2B/36kb=256n^2B/4.5kB=256n^2/4608=ceil(n^2/18)$块
当$n>64\ and\ n<=128$时，由于深度超过了单块bram的极限512，所以矩阵的每8列都需要2块bram。
即便把所有列合在一起申请也是一样，因为为了保证带宽，就没法利用每块里空闲的空间。
所以A和B各需要2n块BRAM。C不变

综上：
A和B各需要的bram数为：
$$ceil(n/64)\times n$$
C需要的bram数为：
$$ceil(n^2/18)$$
按照
$$A+B+C<=[0.9*Total]$$
找到第一次决定的n

#### 新的修改
当n<=64时，按照以上方法进行计算即可
当n>64时：
我们发现，当n从64变为65时，AB需要的bram数会从64变为130。这实际上导致了将近一半的bram空间是空着的，
同时也丢失了对65-128这段空间的搜索
为了更有效地利用空间，采取另一种方式：
对于超过64的部分，单独建立一块乘累加树，这样能够同时支持一块512*512, 以及一块小矩阵
如果能够容纳64 + 64，就能够同时支持两块512*512
所以按照前面所述，为了最大化利用带宽，我们实际上不需要在纵向上拼接Bram，
当bram的深度512全部利用起来了之后，就再开辟一块矩阵即可。
但如果带宽和计算单元足够，则乘累加树宽度需要与计算图中最大矩阵的宽度相同。
举例，如果片上能够容纳2块512x512，则支持的最大矩阵边长为512，且能并行2个512x512
如果能够容纳4块512x512，则支持1024，但仍需拆分为512传输计算
如果能够容纳512x512+256x256，则最大支持512x512，但只能同时计算1个512x512

所以对于类似512x512x4这种情况，如果计算单元足够，则增加并行
如果不够，就纵向垒起来，增加深度


然后根据n判断计算资源是否足够
如果不够，则从n至1遍历，直到找到合适的n

#### 新新的修改
举例，如果A和B均为512，则AB各需要64块bram，C需要256块bram
但事实上，一般的神经网络中很难出现512x512这么大的矩阵
比如resnet18最大只有128，resnet50最大只有256
像vgg这种网络才会有非常大的矩阵，比如vgg11有512，但vgg基本已经淘汰了
所以一般不会计算超大矩阵，
那么，既然不会计算512x512，也就不会有单个矩阵乘完之后直接把C的空间占满
而小矩阵很多时候是要累加的，也就会出现计算了多个小矩阵，结果占用空间仍旧
不是很大的情况，那么，是否可以把分给C的空间分出来一部分，用于提高带宽，
然后再加一组计算单元？
那么这个时候我就需要先切块，根据切块的结果再去分析能不能加计算单元

### 根据简单分配的结果，将计算图拆分为张量计算
现在，我们已经知道了片上支持的最大矩阵，可以按照最大矩阵切分矩阵了。
切分的原则为以下几点：
1. 尽量切大块，但不能超过片上支持的最大边长
2. 矩阵的相乘边(即A的横边和B的纵边)一定为2的幂
3. 矩阵的结果边(即A的纵边和B的横边)的长度要合适，保证填满bram的一行，不够的补0
切分之后，如果不存在片上支持的最大矩阵相乘，则可以酌情削减C的bram分配
#### 怎么削减？
找到一个计算流程，使得其尽可能多累加。使得C的峰值空间占用尽可能少
显然，如果张量计算中存在与芯片支持的最大矩阵相同的矩阵，则无法削减
如果不存在，则初步按照尽可能填满A和B的方式去传输数据和计算，计算C的峰值占用
然后根据C的峰值占用调整资源分配

### 资源的第二次分配
根据上一步计算出来的C的峰值占用，将C始终空闲的部分拿出去给其他部分使用：
如果AB带宽未达到512，则考虑加给AB，但仍需保证AB的带宽为2的幂
如果已经达到或者无法保证AB的带宽为2的幂，则单独成块，需保证新块的带宽为2的幂

如果发现C再多分出去一点，就能使新块的带宽再上一级，或是能使旧块的带宽再上一级，
则计算一下这种更激进的情况下的C的数值，然后和张量表达式生成器商量

否则资源分配到此结束

### 张量表达式的二次分配
在资源的二次分配给出的条件下，寻找新的计算方案
并评估新方案与旧方案相比的损失
根据每一次传输、内存拷贝、片上计算、启动DMA所需的总时间来估计
然后给出结论，是否同意C再多分出去一点

### 资源的三次分配
根据张量表达式二次分配的结果，进行最终分配

### 张量表达式的三次分配
根据资源分配最终结果，切分张量表达式 -->


### 一个想法
突然想到，如果把A读出两行，放入寄存器里面，这样这两行就能同时读取了，然后
分别去和B乘，是否能增加带宽。
1. 如果新增的计算单元的宽度不足以达到bram的宽度，那么A的第一行和A的第二行
所能够使用的计算单元的数量不同，那么它们和B计算的速度就不同，读取B的速度也就不同。
而B受带宽限制，每周期只能读一行，无法同时供给A的两行，所以不行
2. 如果新增的计算单元的宽度能够达到bram的宽度，则A的两行读取B的速度相同，B读出一行
可以同时供给A的两行使用，计算速度翻倍



### Instruction width alloction
Since the side length of divided matrix are all power of 2, we can record
them by the exponent, and in this way we need less bits.


### How to design top generator for CONVOLUTION?
Since the max matrix side length we have is no more than `max_len_support`, 
and `max_len_support` is equal to the length of a bram line, we should read `n`
bram_A lines per calc when `mult_side_len` is equal to `max_len_support` and 
`calc_unit_per_bram_group` is equal to `n`. At the same time, we should read 
1 bram_B line per cycle.
Since the min matrix side length we have is no less than `min_len_support`, 
and `min_len_support` is no less than sqrt of `max_len_support`, which means 
that a minimum matrix can fill at least 1 bram line, we should also read 1
bram_B line per cycle, and 1 bram_A line per calc if `mult_side_len`*
`calc_unit_per_bram_group` is no more than `max_len_support`, or more than 1 
lines if the product is more.
So we do not need to care how many bram_B lines we should read per cycle, since 
it is always 1. What we should care is how many bram_A lines we should read per 
calc. 
So, how many bram_A lines should we read per calc? 
- if `mult_side_len`*`calc_unit_per_bram_group` <= `max_len_support`, 1 per calc
- else, `mult_side_len`*`calc_unit_per_bram_group` // `max_len_support` per calc

#### Total lines of bram that mat A and B cost
Matrix A has a left side with len `A_left_side_len`
Matrix A has a up side with len `mult_side_len`
Matrix B has a left side with len `mult_side_len`
Matrix B has a up side with len `B_up_side_len`
So bram lines A cost is `A_left_side_len`*`mult_side_len`//`max_len_support`
Bram lines B cost is `B_up_side_len`*`mult_side_len`//`max_len_support`
Besides, we should notice that, since the side lengths are all power of 2,
if calc_unit is more than 1, and lines of bram that mat A cost is more than 
calc_unit, the lines of bram it cost must be multiple of calc_unit(or must
be power of 2), so we do not worry about the situation that when reading mat A,
we have to read lines of next mat.

#### What is the latency of conv module?
`2 + log2(accumulate_length)` cycles.
For example, 10 cycles for add256.

#### What should we do to calculate one convolution?
1. Use `count0` to set `bram_A_addr` and `bram_B_addr` 
Increase `count0` each cycle
If `mult_side_len`*`calc_unit_per_bram_group`//`max_len_support` is more than 1,
pause `count0` for a several cycles after a whole loop of bram_B, and also pause 
at the beginning. 
How many cycles? `mult_side_len`*`calc_unit_per_bram_group`//`max_len_support`-1 
Use `count3` to count pause cycles
2. Use `count1` to select `bram_A_dout`
If we read 1 bram A line in each B loop, we do not pause `count0`, and increase 
`count1` since `count0==2`. And we use `count1` to select `bram_A_dout`.
If we read more than 1 bram A lines in each B loop, we do pause `count0`, and we
do not need to select `bram_A_dout` now, since they are all used. So `count1` is 
useless in this case.
So, increase `count1` since `count0==2` if read 1 bram_A line per B loop
otherwise ignore `count1`
3. `bram_B_dout` do not need to select, it is totally used
4. Use `count2` to select `bram_r_we`
Increase `count2` each cycle, since write first result into bram_r.
`count2` should pause after a loop. Should not pause `count2` together 
with `count0`, since conv module do not stall when `count0` stall. So stall
`count2` when finished a loop, and count the stall cycle with `count4`
5. Use `count5` to set `bram_r_addr`
Increase `count5` each cycle, since read first result from bram_r.
`count5` should pause after a loop. Should not pause `count5` together
with `count0`, since conv module do not stall when `count0` stall. So stall
`count5` when finished a loop, and count the stall cycle with `count6`
5. Set `bram_a_addr` with `count0` and `count3`
When `count0==0` and `count3==0`, set `bram_a_addr` to `pl_weight_start_addr_conv`,
If `mult_side_len`*`calc_unit_per_bram_group`//`max_len_support` is more than 1,
stall `count0` and increase `count3`, and increase `bram_a_addr` together, and
read `bram_a_dout` into `temp` synchronously;
else if read 1 bram A line per loop, do not need to stall `count0`
6. Set `bram_b_addr` with `count0`
When `count0[x:0]==0`, set `bram_b_addr` to `pl_feature_map_addr_conv`, where `x`
is decided by total bram_b line used. Otherwise, increase `bram_b_addr` each cycle.
7. Set `temp_w` with `count0`, `count1`, `count3`
If read 1 bram_a line per loop, use `count1` to select `bram_a_dout`.
Else, use `count3` to select `temp_w`
8. Set `temp_f` with `count0`
Write `bram_b_dout` into `temp_f` each cycle.
9. Set `conv_in` with `count0`
Write `temp_w` into `conv_ina`
Write `temp_f` into `conv_inb`
10. Set `bram_r_addrb` with `count5` and `count6`
If `pl_save_type_conv==1`, 
set `bram_r_addrb` with `pl_output_start_addr_conv` at the beginning, and 
increase per `n` cycles.
`n` = ?
11. Set `bram_r_addra` with `count2` and `count4`
set `bram_r_addra` with `pl_output_start_addr_conv` at the beginning, and 
increase per `n` cycles.
12. Set `bram_r_wea` with `count2`
Set part of `bram_r_wea` to 1 and others to 0
13. Set `bram_r_dina` with `count2`
If `pl_save_type_conv==0`, save conv module output into `bram_r_dina`
else, add `bram_r_doutb` and save into `bram_r_dina`
14. Stop 
After calculation finished, reset `count`, `bram_r_wea` and branch to other states
When to stop? 
For example, assume that calc_unit is 4, bram_A line is 16, bram_B line is 16,
max_len_support is 256. 
How many A lines? `pl_mat_A_line`
How many B lines? `pl_mat_B_line`
How many loops per A line(if use no more than 1 A line per loop)?
    `max_len_support // mult_side_len // calc_unit`(
        if `mult_side_len * calc_unit <= max_len_support`
    )
How many A lines per loop(if use more than 1 A line per loop)?
    `mult_side_len * calc_unit // max_len_support`(
        if `mult_side_len * calc_unit > max_len_support`
    )
So the stop cycle count should be:
`pl_mat_A_line * pl_mat_B_line * loops_per_A_line` if use no more than 1 A line per loop
`pl_mat_A_line * pl_mat_B_line // A_lines_per_loop` if use more than 1 A line per loop
Now, we got `count_finish_cycle`, which means we should `count_finish_cycle` 
times of calc in one convolution



# Attention
About write_back, if calculation unit is more than 1, such as 2, then we will do 
post process in two parallel modules, and the results return in two streams. 
The result streams will store into bram_r in two groups of bram_r cols, and so,
when we write back the results, the results are not contiguous.
For example, the results in bram_r may look like the below:
r00 r01 r10 r11 
r02 r03 r12 r13
r04 r05 r14 r15
r06 r07 r16 r17
r20 r21 r30 r31
...     ...
In order to save the dma startup time, we would like to transfer all post processed
results in one transfer, and in this way, we cannot distinguish the mult_side_len, 
and will not know when to change bram_r_addr to transfer in matrix order, so we 
have to transfer the data in bram_r order, and it does not match the matrix order.
So, we have to convert it back to matrix order in C program.