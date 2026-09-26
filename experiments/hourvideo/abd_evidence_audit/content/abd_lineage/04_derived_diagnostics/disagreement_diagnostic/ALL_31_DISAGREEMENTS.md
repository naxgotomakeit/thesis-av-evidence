# ABD evidence-audit disagreement diagnosis (no gold)

This report preserves both rounds' labels. It does not adjudicate or create a new label.

## 1. E0008: `partially_supported` → `unsupported`

- 旧理由：The inspected food-preparation map shows oven tray handling followed by pizza/bread cutting and brushing, broadly matching option B, but the exact utensil and substance are uncertain.
- 旧证据引用：`["map:C17", "map:C18", "map:C19", "map:C20"]`
- 新理由：The map shows pizza/tray handling but no bread paddle, brush, or curry-like application after the referenced oven event, so the selected multi-part activity is not supported.
- 新证据引用：`["map:C18-C20"]`
- 具体分歧：旧审把托盘/披萨相关线索视为对复合选项的部分支持，新审认为面包铲、刷子和所涂物质三个关键条件均未出现，因而降为无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（0张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 2. E0021: `supported` → `contradicted`

- 旧理由：Map intervals and inspected food-preparation frames show cucumber/radish washing and preparation lasting substantially longer than the brief pot-stirring interval.
- 旧证据引用：`["image:frame_00000.jpg@0.0s", "image:frame_00045.jpg@45.0s", "image:frame_00090.jpg@90.0s", "image:frame_00315.jpg@315.0s", "image:frame_00630.jpg@630.0s", "image:frame_00945.jpg@945.0s", "image:frame_01800.jpg@1800.0s", "image:frame_02430.jpg@2430.0s", "image:frame_02475.jpg@2475.0s", "map_locator:<SOURCE_WORKSPACE>/outputs/abd_eval300_evidence_audit_v1/review_packages/E0021.map.json"]`
- 新理由：The selected statement refers to white radish although the evidence identifies cucumber, and cooking/stirring occurs repeatedly across the map beyond the vegetable-preparation interval, so the claimed duration ordering is not supported.
- 新证据引用：`["image:frames_001-009", "map:C01-C03,C05,C07,C14-C16,C19"]`
- 具体分歧：旧审把黄瓜/萝卜准备与短暂搅拌作时长比较并接受选项，新审同时认为食材身份不符且地图还有多段搅拌，故判为明确冲突。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 3. E0023: `unsupported` → `contradicted`

- 旧理由：Inspected cooking frames show vegetables/ingredients being added, but do not identify tomato sauce; the selected ingredient is not verifiable.
- 旧证据引用：`["image:frame_00585.jpg@585.0s", "image:frame_00630.jpg@630.0s", "image:frame_00675.jpg@675.0s", "image:frame_00900.jpg@900.0s", "image:frame_00945.jpg@945.0s", "image:frame_00990.jpg@990.0s", "image:frame_01575.jpg@1575.0s", "image:frame_01620.jpg@1620.0s", "map_locator:<SOURCE_WORKSPACE>/outputs/abd_eval300_evidence_audit_v1/review_packages/E0023.map.json"]`
- 新理由：After the mushroom-preparation sequence, the map explicitly shows water being poured into the pot and stirred; neither the frames nor map establish tomato sauce as the added ingredient.
- 新证据引用：`["image:frames_004-007", "map:C06(900-1035s)"]`
- 具体分歧：旧审只认为番茄酱无法确认，新审把相关时段地图中的“加水并搅拌”视为对番茄酱这一具体选择的正面反证。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（8张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 4. E0061: `supported` → `contradicted`

- 旧理由：Inspected kitchen/house frames show rinsing followed by leaving-preparation activity rather than dish storage.
- 旧证据引用：`["image:frame_01402.jpg@1402.0s", "image:frame_01417.jpg@1417.0s", "image:frame_01432.jpg@1432.0s", "image:frame_01440.jpg@1440.0s", "image:frame_01447.jpg@1447.0s", "image:frame_01462.jpg@1462.0s", "image:frame_01477.jpg@1477.0s", "image:frame_01485.jpg@1485.0s", "image:frame_01492.jpg@1492.0s", "map_locator:<SOURCE_WORKSPACE>/outputs/abd_eval300_evidence_audit_v1/review_packages/E0061.map.json"]`
- 新理由：After the rinsing/washing sequence, the map and frames show the wearer walking around the room and looking at the television before beginning to prepare to leave, so 'preparing to leave the house' is not the immediate next event.
- 新证据引用：`["map:C18-C20", "image:frames_004-009@1440-1492s"]`
- 具体分歧：旧审把冲洗后的后续活动概括为准备离开，新审认定其间明确插入了走动和看电视，因此“紧接着准备离开”被时序反驳。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 5. E0072: `partially_supported` → `unsupported`

- 旧理由：Available map and sampled frames provide related context, but the selected claim is not fully established for every required detail.
- 旧证据引用：`["map_locator:<SOURCE_WORKSPACE>/outputs/abd_eval300_evidence_audit_v1/review_packages/E0072.map.json"]`
- 新理由：The map contains several push-stick and pen interactions, but it does not cleanly establish that the push stick was used more often than the pen; the captioned event counts are not sufficient for that frequency claim.
- 新证据引用：`["map:C01,C03,C04,C08,C09"]`
- 具体分歧：旧审把推杆和笔的若干相关事件保留为部分支持，新审认为这些粗粒度事件不足以完成两者全程频次比较。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（0张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 6. E0077: `partially_supported` → `unsupported`

- 旧理由：Inspected evidence provides related context, but does not fully establish every detail of the selected claim.
- 旧证据引用：`["image:frame_00180.jpg@180.0s", "image:frame_00600.jpg@600.0s", "image:frame_01200.jpg@1200.0s", "image:frame_01665.jpg@1665.0s", "image:frame_01700.jpg@1700.0s", "image:frame_02500.jpg@2500.0s", "image:frame_03800.jpg@3800.0s", "image:frame_03950.jpg@3950.0s", "image:frame_04000.jpg@4000.0s"]`
- 新理由：The sampled bakery frames show dough/bread preparation and trays, but they cannot establish the selected exhaustive list of fifteen distinct food items, many of which are not identifiable in the provided images.
- 新证据引用：`["image:frames_001-009@1180-4000s"]`
- 具体分歧：旧审把面包制作背景当作长清单的部分支持，新审要求逐项识别十五种食物，认为现有九帧不能支持该穷举选项。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 7. E0112: `supported` → `contradicted`

- 旧理由：Inspected home frames show prerequisite room movement and item pickup before vacuuming, consistent with the map sequence.
- 旧证据引用：`["image:frame_00045.jpg@45.0s", "image:frame_00090.jpg@90.0s", "image:frame_00135.jpg@135.0s", "image:frame_00180.jpg@180.0s", "image:frame_00225.jpg@225.0s", "image:frame_00270.jpg@270.0s", "image:frame_00360.jpg@360.0s", "image:frame_00405.jpg@405.0s", "image:frame_00450.jpg@450.0s", "map_locator:<SOURCE_WORKSPACE>/outputs/abd_eval300_evidence_audit_v1/review_packages/E0112.map.json"]`
- 新理由：Before vacuuming, the map and frames show room inspection plus handling a bag and pillow; they do not show carrying a table and instead show different items than the selected table/shoe/dog-chain prerequisite sequence.
- 新证据引用：`["image:frames_001-009@45-450s", "map:C02-C05"]`
- 具体分歧：旧审认为吸尘前的房间移动和取物大体吻合，新审指出实际物品是包和枕头而非桌子、鞋和狗链，故认为前置序列冲突。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 8. E0113: `supported` → `unsupported`

- 旧理由：All nine inspected frames include kitchen-to-hall/living-room views, with the doorway/passage connecting the spaces, supporting the selected navigation route.
- 旧证据引用：`["image:frame_00630.jpg@630.0s", "image:frame_00652.jpg@652.0s", "image:frame_00675.jpg@675.0s", "image:frame_00720.jpg@720.0s", "image:frame_00787.jpg@787.0s", "image:frame_00810.jpg@810.0s", "image:frame_02250.jpg@2250.0s", "image:frame_02295.jpg@2295.0s", "image:frame_02475.jpg@2475.0s"]`
- 新理由：The frames show a physical route between kitchen and living room, but the selected option is only an opaque navigation-image path and the corresponding option image is not included, so the evidence cannot establish that option C depicts that route.
- 新证据引用：`["image:frames_001-009@630-2475s", "map:C06-C17"]`
- 具体分歧：旧审凭帧中可见的厨房到客厅通路接受选项，新审要求验证不在包内的选项C导航图本身，因无法把路径对应到该图而判无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`operational_definition_difference_missing_option_image`；该归类不裁决哪轮正确。

## 9. E0114: `partially_supported` → `unsupported`

- 旧理由：Inspected frames show white-radish preparation and later pan cooking/stirring contexts, but the supplied sample does not provide continuous timing sufficient to prove the duration comparison.
- 旧证据引用：`["image:frame_00007.jpg@7.0s", "image:frame_00022.jpg@22.0s", "image:frame_00037.jpg@37.0s", "image:frame_00052.jpg@52.0s", "image:frame_00142.jpg@142.0s", "image:frame_00172.jpg@172.0s", "image:frame_01642.jpg@1642.0s", "image:frame_01657.jpg@1657.0s", "image:frame_01800.jpg@1800.0s"]`
- 新理由：The sparse frames show both vegetable preparation and later cooking/stirring, but they do not provide continuous event boundaries sufficient to compare total time spent on the two activities.
- 新证据引用：`["image:frames_001-009@7-1800s"]`
- 具体分歧：两轮都认为稀疏帧不足以完成时长比较，分歧仅在把相关活动可见性记作部分支持，还是因缺少完整边界直接记作无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 10. E0122: `supported` → `unsupported`

- 旧理由：Inspected cooking frames show a spoon/utensil used to tap or scrape excess food while cooking.
- 旧证据引用：`["image:frame_00585.jpg@585.0s", "image:frame_00607.jpg@607.0s", "image:frame_00630.jpg@630.0s", "image:frame_00697.jpg@697.0s", "image:frame_00720.jpg@720.0s", "image:frame_00765.jpg@765.0s", "image:frame_00787.jpg@787.0s", "image:frame_00945.jpg@945.0s", "image:frame_01350.jpg@1350.0s"]`
- 新理由：The cooking frames show several utensils in use, but the tool used for the specific tap-off-excess-food action is not clearly identifiable as tongs.
- 新证据引用：`["image:frames_001-009@585-1350s"]`
- 具体分歧：旧审把画面中的勺状器具视为执行敲落动作的工具，新审认为具体动作及工具均看不清，尤其不能确认是选项所称的夹子。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 11. E0269: `supported` → `partially_supported`

- 旧理由：The map shows mallet/hammer tapping as a recurring brick-laying action and, after the 2200-2400s brick sequence, another hammer-tapping event follows at 2475-2520s. This grounds the prediction that continued brickwork would next involve hitting bricks with a mallet.
- 旧证据引用：`["map M050-M054, 2205.000-2430.000s: mallet use, brick placement and alignment", "map M056, 2475.000-2520.000s: brick placed and hit with hammer", "frames 2200.000-2400.000s: continuing brickwork"]`
- 新理由：Repeated mallet/hammer brick-adjustment actions make the selected continuation plausible within the work pattern, but the late sampled frames do not establish it as the actual next event after the queried anchor.
- 新证据引用：`["image:frames_001-009", "map:C02-C20"]`
- 具体分歧：旧审用反复敲砖模式及稍后事件支持“下一步仍敲砖”，新审要求锚点后紧邻画面，认为重复模式只能让预测合理而不能证明实际下一事件。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（7张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 12. E0289: `unsupported` → `partially_supported`

- 旧理由：The sparse frames confirm woodworking operations and a late cleanup-like interval, but snapshots at widely separated times do not establish the durations of final cutting/sorting versus workshop cleanup. The predicted duration comparison cannot be supported from these frames alone.
- 旧证据引用：`["frames 495.000-3285.000s: measuring, sawing, and vacuum-related workshop actions", "frame 4044.000s: late workshop work/cleanup context"]`
- 新理由：The woodworking frames span a long period of cutting/sorting planks and a shorter sampled cleanup/vacuum interval, which is consistent with the selected duration direction, but sparse frames do not establish complete event durations.
- 新证据引用：`["image:frames_001-008@0-4040s"]`
- 具体分歧：两轮都承认缺少连续时长边界；旧审因此判无支持，新审把木工作业覆盖更广、清理采样更短视为方向性线索而给部分支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（8张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 13. E0311: `partially_supported` → `supported`

- 旧理由：The opened frames and corresponding map regions provide relevant cooking context, but the selected frequency/positional comparison is not fully established by the sampled evidence.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "map:C05", "map:C06", "map:C07", "map:C08", "map:C09", "map:C10", "map:C11", "map:C12", "map:C13", "map:C14", "map:C15", "map:C16", "map:C17", "map:C18", "map:C19", "map:C20", "map:C21", "map:C22", "image:frame_00037.jpg@37.0s", "image:frame_00067.jpg@67.0s", "image:frame_00142.jpg@142.0s", "image:frame_00157.jpg@157.0s", "image:frame_00247.jpg@247.0s", "image:frame_00382.jpg@382.0s", "image:frame_00472.jpg@472.0s", "image:frame_00487.jpg@487.0s", "image:frame_00652.jpg@652.0s", "image:frame_00922.jpg@922.0s", "image:frame_01147.jpg@1147.0s", "image:frame_01642.jpg@1642.0s", "image:frame_01957.jpg@1957.0s", "image:frame_02002.jpg@2002.0s", "image:frame_02317.jpg@2317.0s", "image:frame_02407.jpg@2407.0s"]`
- 新理由：Across the map, knife use occurs repeatedly for peeling/cutting cucumber, coconut, and tomatoes in more distinct preparation intervals than spoon use for stirring; the sampled frames are consistent with this pattern.
- 新证据引用：`["map:C01,C03,C08,C10,C17,C20-C21 versus C02,C05,C07,C14-C16,C19", "image:frames_001-016"]`
- 具体分歧：旧审认为刀和勺的频次比较未被完整计数，新审按地图中分散出现的操作区间计数，认为刀的出现区间足以区分。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（16张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 14. E0330: `partially_supported` → `contradicted`

- 旧理由：Opened sampled frames and the associated map provide related context, but the decisive claim is not fully established across the video.
- 旧证据引用：`["map:C12", "image:frame_01582.jpg@1582.0s", "image:frame_01717.jpg@1717.0s", "image:frame_01732.jpg@1732.0s", "image:frame_01747.jpg@1747.0s"]`
- 新理由：The late frames place the cooking pot on the cooker/stove beside the sink area, and the map explicitly says the pot is moved from the sink to the cooker rather than being placed on the countertop as selected.
- 新证据引用：`["image:frames_001-004@1582-1747s", "map:C12"]`
- 具体分歧：旧审只称锅具放置证据不完整，新审读取C12为从水槽移到灶台，认为这明确反驳“放到台面”。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（4张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 15. E0339: `partially_supported` → `unsupported`

- 旧理由：Opened sampled frames and map regions provide related context, but the decisive claim is not fully established across the video.
- 旧证据引用：`["map:C08", "map:C09", "map:C10", "map:C11", "image:frame_00765.jpg@765.0s", "image:frame_00820.jpg@820.0s", "image:frame_00855.jpg@855.0s", "image:frame_00877.jpg@877.0s", "image:frame_00900.jpg@900.0s", "image:frame_01000.jpg@1000.0s", "image:frame_01050.jpg@1050.0s", "image:frame_01057.jpg@1057.0s", "image:frame_01070.jpg@1070.0s", "image:frame_01150.jpg@1150.0s", "image:frame_01160.jpg@1160.0s", "image:frame_01200.jpg@1200.0s", "image:frame_01300.jpg@1300.0s", "image:frame_01500.jpg@1500.0s", "image:frame_01507.jpg@1507.0s", "image:frame_01520.jpg@1520.0s"]`
- 新理由：The frames show a kitchen/pantry with assorted products, but neither the images nor map uniquely establish the selected exhaustive list of frozen food, soda, beans, chili powder, tomato paste, and cheese as the foods handled.
- 新证据引用：`["image:frames_001-016@765-1520s", "map:C08-C12"]`
- 具体分歧：旧审把厨房/食品背景算作长食材清单的部分支持，新审要求穷举清单中每种食材均被清楚识别和处理，故判无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（16张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 16. E0351: `partially_supported` → `contradicted`

- 旧理由：Opened sampled frames and map regions provide related context, but the decisive claim is not fully established.
- 旧证据引用：`["map:C12", "map:C13", "map:C14", "map:C15", "map:C16", "map:C17", "map:C18", "map:C19", "map:C20", "map:C21", "map:C22", "map:C23", "map:C24", "image:frame_01125.jpg@1125.0s", "image:frame_01395.jpg@1395.0s", "image:frame_01440.jpg@1440.0s", "image:frame_01485.jpg@1485.0s", "image:frame_01530.jpg@1530.0s", "image:frame_01620.jpg@1620.0s", "image:frame_01665.jpg@1665.0s", "image:frame_01755.jpg@1755.0s", "image:frame_01800.jpg@1800.0s", "image:frame_01890.jpg@1890.0s", "image:frame_01980.jpg@1980.0s", "image:frame_02025.jpg@2025.0s", "image:frame_02115.jpg@2115.0s", "image:frame_02250.jpg@2250.0s", "image:frame_02430.jpg@2430.0s", "image:frame_02475.jpg@2475.0s"]`
- 新理由：After the late tomato-cutting/salt/blender steps, the map proceeds to walking and then prolonged phone use; the selected six-step sequence largely describes unrelated cooking/cleaning events that occur earlier or are not established after the tomato event.
- 新证据引用：`["image:frames_012-016@2025-2475s", "map:C20-C24"]`
- 具体分歧：旧审把后番茄阶段仅视为证据不足，新审按晚段地图认定随后是走动和长时间用手机，与所选六步烹饪/清洁序列冲突。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（16张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 17. E0394: `partially_supported` → `contradicted`

- 旧理由：Opened submitted frames and map regions provide related context, but the decisive claim is not fully established across the video.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "map:C05", "map:C06", "map:C07", "map:C08", "map:C09", "map:C10", "map:C11", "map:C12", "map:C13", "map:C14", "map:C15", "map:C16", "map:C17", "map:C18", "map:C19", "map:C20", "map:C21", "map:C22", "map:C23", "map:C24", "image:frame_00630.jpg@630.0s", "image:frame_00675.jpg@675.0s", "image:frame_00810.jpg@810.0s", "image:frame_00900.jpg@900.0s", "image:frame_01125.jpg@1125.0s", "image:frame_01575.jpg@1575.0s", "image:frame_01620.jpg@1620.0s", "image:frame_01800.jpg@1800.0s", "image:frame_01980.jpg@1980.0s", "image:frame_02025.jpg@2025.0s", "image:frame_02070.jpg@2070.0s", "image:frame_02115.jpg@2115.0s", "image:frame_02250.jpg@2250.0s"]`
- 新理由：The provided frames and map place the coconut activity around 630-675s before the water addition around 900s; no later coconut-grating event is established, so the selected 'after water ... before grating' relation is temporally contradicted.
- 新证据引用：`["image:frames_001-004@630-900s", "map:C07-C10"]`
- 具体分歧：旧审未对关键先后作确定判断，新审认定椰子处理发生在加水之前，直接反驳选项所称“加水后、磨椰子前”。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（13张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 18. E0400: `partially_supported` → `unsupported`

- 旧理由：Opened sampled frames and map regions provide related context, but decisive claim is not fully established.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "map:C05", "map:C06", "map:C07", "map:C08", "map:C09", "image:frame_00000.jpg@0.0s", "image:frame_00225.jpg@225.0s", "image:frame_00405.jpg@405.0s", "image:frame_01170.jpg@1170.0s", "image:frame_01305.jpg@1305.0s", "image:frame_01575.jpg@1575.0s", "image:frame_01620.jpg@1620.0s"]`
- 新理由：The frames and map show several vegetables being prepared and cooked, but they do not establish an exact total of four vegetable types added to the stew.
- 新证据引用：`["image:frames_001-009", "map:C01-C09"]`
- 具体分歧：旧审把多种蔬菜准备背景视为部分线索，新审认为无法得到“恰好四种加入炖菜”的完整可计数记录。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（7张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 19. E0440: `unsupported` → `partially_supported`

- 旧理由：The readable map C01–C24 describes food preparation, cleaning and garden activity but never mentions gloves or their color. The transparent-glove choice therefore has no supporting evidence.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "map:C05", "map:C06", "map:C07", "map:C08", "map:C09", "map:C10", "map:C11", "map:C12", "map:C13", "map:C14", "map:C15", "map:C16", "map:C17", "map:C18", "map:C19", "map:C20"]`
- 新理由：The map shows transparent plastic bags being placed/adjusted on the hands before going outside, which is consistent with transparent makeshift gloves, but it does not explicitly identify them as gloves.
- 新证据引用：`["map:C20(1485-1595s)"]`
- 具体分歧：旧审认为地图没写手套或颜色所以无支持，新审把手上透明塑料袋解释为透明临时手套，因名称未明确而给部分支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（0张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 20. E0448: `supported` → `partially_supported`

- 旧理由：The mapped spray-paint organization/painting sequence spans C15-C18 (1800-3240s), whereas canopy setup is confined to C20 (3465-3720s). Even allowing for coarse region boundaries, the former interval is substantially longer, directly supporting option D.
- 旧证据引用：`["map C15-C18 1800-3240s: spray-can handling, organizing paint supplies, and repeated painting", "map C20 3465-3720s: unfolds tent/canopy and handles its frame, poles, and fabric"]`
- 新理由：The mapped spray-paint/painting sequence spans substantially longer than the canopy-setup sequence, but the map does not clearly establish staircase painting or the woman’s participation.
- 新证据引用：`["map:C15-C18(1800-3240s)", "map:C20(3465-3720s)"]`
- 具体分歧：旧审只按两个地图区间的明显长度差支持选项，新审认为选项还限定楼梯及女性共同参与，而这些条件未被明确建立。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（0张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 21. E0460: `unsupported` → `contradicted`

- 旧理由：The sampled frames show sausage and eggs already cooking, later a cabinet interaction, and then movement near the sink. They do not show the egg tray being put back, cooking oil being used, or sausage being cut on a board in the selected B order.
- 旧证据引用：`["frames 360-450s: sausage and eggs cooking in pan", "frame 495s: cabinet interaction", "frame 540s: cooked eggs/sausage in pan", "frames 585-630s: sink-area movement"]`
- 新理由：Immediately after the egg-related frames, the visible sequence shows pan/egg handling and plate/cabinet activity, not the selected order of returning the egg tray, using cooking oil, and then cutting sausage.
- 新证据引用：`["image:frames_001-007@360-630s"]`
- 具体分歧：旧审认为三步动作及顺序没有出现，新审进一步把可见的锅/蛋、盘子和橱柜序列当作与所选三步顺序相冲突。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（7张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 22. E0489: `partially_supported` → `unsupported`

- 旧理由：The frames directly show an open laptop, motherboard/system board, CPU area, cooling fans/thermal assembly, hard drive, circuit boards/connectors, and later keyboard and mouse interaction. They do not visibly establish every item in option A's long exhaustive list—particularly interaction with a camera, flash drive, memory card, and several specifically named subassemblies—so the complete list is only partially supported.
- 旧证据引用：`["frames 682-1147 s: laptop internals, boards, CPU area and connectors handled", "frames 2392-2752 s: keyboard/mouse use and reassembled laptop internals including hard drive/cooling"]`
- 新理由：The laptop frames show several internal components and peripherals, but they do not establish every item in the selected exhaustive technology-object list, including several specifically named boards/connectors/devices.
- 新证据引用：`["image:frames_001-012@7-2752s"]`
- 具体分歧：两轮都认为只看到长技术物件清单的一部分；旧审计为部分支持，新审因穷举条件未满足而计为无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（12张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 23. E0532: `unsupported` → `contradicted`

- 旧理由：The map's coarse intervals heavily intermix digging, alignment, brick movement, and cement work, so their durations cannot be separated by simply summing entire regions. It also never identifies a combined hoe-and-fork. Consequently the claimed comparison that digging/alignment took longer than the broader brick-and-cement construction work is not supported by the available timing evidence.
- 旧证据引用：`["map C02-C17 45-3060s: overlapping brick placement, alignment, cement, and digging actions", "map C16 2610-2835s: pickaxe/spade digging, not combined hoe-and-fork"]`
- 新理由：The map contains extensive repeated brick/cement handling across many regions, while digging/alignment events are more limited; the selected claim that digging/alignment lasted longer is not supported and trends opposite to the visible coverage.
- 新证据引用：`["map:C02-C20"]`
- 具体分歧：旧审认为活动混杂所以无法比较时长，新审从地图覆盖量判断砖/水泥活动明显更多，认为所选方向呈相反趋势。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（0张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 24. E0557: `contradicted` → `partially_supported`

- 旧理由：Knife use occurs in several preparation blocks, but the map records repeated spoon/cooking-stick stirring throughout C05-C09, spanning more repeated cooking actions. This conflicts with the prediction that the knife was used more frequently during cooking.
- 旧证据引用：`["map C01-C05, 0.000-810.000s: knife preparation actions", "map C05-C09, 585.000-1675.000s: repeated stirring with spoon/cooking stick"]`
- 新理由：Knife use and wooden-spoon use both recur throughout the map, with knife-related preparation appearing somewhat more often, but the coarse summaries do not support a reliable event-frequency comparison for the exact 'cooking stick' category.
- 新证据引用：`["map:C01-C09"]`
- 具体分歧：旧审认为反复搅拌多于刀具使用而反驳选项，新审认为刀具相关区间可能更多但类别和计数不清，只保留部分支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（0张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 25. E0657: `supported` → `partially_supported`

- 旧理由：Frames show hands opening and handling laptop internal components, supporting the selected technology interaction statement.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "image:frame_00100.jpg@100.0s", "image:frame_00500.jpg@500.0s", "image:frame_00810.jpg@810.0s", "image:frame_00855.jpg@855.0s", "image:frame_00945.jpg@945.0s", "image:frame_00990.jpg@990.0s", "image:frame_01000.jpg@1000.0s", "image:frame_01620.jpg@1620.0s", "image:frame_02205.jpg@2205.0s"]`
- 新理由：The map and frames show prolonged handling of laptop internals and components, with only limited cable/connector handling; they do not cleanly establish the selected comparison between the named component-handling category and wire-connection category.
- 新证据引用：`["map:C02-C04", "image:frames_002-009@500-2205s"]`
- 具体分歧：旧审把长时间处理电脑内部部件视为足以支持，新审认为题目要求与接线活动作比较，而两类事件边界不足以可靠比较。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 26. E0666: `partially_supported` → `supported`

- 旧理由：Sampled frames and map regions provide related activity context, but do not fully establish every detail of the selected option.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "map:C05", "map:C06", "map:C07", "map:C08", "map:C09", "map:C10", "map:C11", "map:C12", "map:C13", "map:C14", "map:C15", "map:C16", "map:C17", "map:C18", "map:C19", "map:C20", "image:frame_01440.jpg@1440.0s", "image:frame_01460.jpg@1460.0s", "image:frame_01480.jpg@1480.0s", "image:frame_01485.jpg@1485.0s", "image:frame_01492.jpg@1492.0s", "image:frame_01507.jpg@1507.0s", "image:frame_01530.jpg@1530.0s", "image:frame_01575.jpg@1575.0s", "image:frame_01582.jpg@1582.0s", "image:frame_01592.jpg@1592.0s", "image:frame_01594.jpg@1594.0s"]`
- 新理由：The provided frames visibly show transparent hand coverings/gloves on the camera wearer immediately before and while outside, directly supporting the selected color.
- 新证据引用：`["image:frames_008-011@1575-1594s", "map:C20"]`
- 具体分歧：旧审笼统认为透明手套颜色细节未完全建立，新审在1575–1594秒帧中直接识别到透明手部覆盖物，因而升为支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（11张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 27. E0705: `unsupported` → `supported`

- 旧理由：The map explicitly documents knife/spoon use and later sink-cleaning activity, but does not state that either item was or was not cleaned; the selected “neither” conclusion is not supported.
- 旧证据引用：`["map:C01", "map:C02", "map:C03", "map:C04", "map:C05", "map:C06", "map:C07", "map:C08", "map:C09", "map:C10", "map:C11", "map:C12", "map:C13", "map:C14", "map:C15", "map:C16", "map:C17", "map:C18", "map:C19", "map:C20", "map:C21", "map:C22", "map:C23", "map:C24"]`
- 新理由：During the documented sink-cleaning sequence the map explicitly lists washing plates/pans but only putting the knife in the sink and picking it up; neither knife nor wooden spoon is shown being washed, and the sampled sink frames are consistent with that account.
- 新证据引用：`["image:frames_005-009@1800-2017s", "map:C13"]`
- 具体分歧：旧审遵循“未提及清洗不能证明没洗”而判无支持，新审认为清洗清单明确列出盘/锅却只写刀的放取，足以支持刀和木勺均未清洗。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 28. E0756: `partially_supported` → `supported`

- 旧理由：The coconut-processing frames and map positively show repeated knife cuts on coconut between roughly 585 and 675 seconds. They do not provide an exhaustive count of coconut-grater use, so repeated knife use alone cannot establish that the knife was used more frequently than the grater.
- 旧证据引用：`["map:C07", "map:C08", "map:C09", "frame:517.000", "frame:532.000", "frame:540.000", "frame:547.000", "frame:560.000", "frame:580.000", "frame:630.000", "frame:650.000", "frame:670.000"]`
- 新理由：The coconut-related map and frames repeatedly show a knife being used to cut the coconut, while no coconut-grater use is established in the relevant evidence, supporting the selected frequency direction.
- 新证据引用：`["image:frames_001-009@517-670s", "map:C07-C09"]`
- 具体分歧：旧审认为看见多次用刀但没完整统计磨椰器，不能完成频次比较；新审把反复用刀且相关证据未出现磨椰器视为足够支持方向。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`evidence_interpretation_difference`；该归类不裁决哪轮正确。

## 29. E0825: `partially_supported` → `unsupported`

- 旧理由：The frames clearly show distinct industrial/electrical work areas, including an outdoor equipment area, control panel area and wooden storage-like area. They support multiple locations but do not provide reliable labels proving the exact exhaustive set Garage, Construction Site, and Warehouse.
- 旧证据引用：`["frame:7.000", "frame:37.000", "frame:300.000", "frame:1000.000", "frame:1150.000", "frame:1400.000"]`
- 新理由：The industrial/construction frames show electrical/control-panel and construction work but do not distinctly establish three unique locations labeled Garage, Construction Site, and Warehouse.
- 新证据引用：`["image:frames_001-006@7-1400s"]`
- 具体分歧：两轮都只确认多个工业/施工场景而不能明确区分车库、工地、仓库；旧审给部分支持，新审因三地点穷举未成立而给无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（6张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 30. E0856: `partially_supported` → `unsupported`

- 旧理由：The submitted frames visibly support several construction items in the selected list, including gloves, bricks, mallet/hammer, bucket, string line, tape measure and powered cutting equipment. Nine snapshots cannot validate the exhaustive 25-item inventory or distinguish mattock from the neighboring long-handled tools.
- 旧证据引用：`["frame:7.000", "frame:157.000", "frame:472.000", "frame:517.000", "frame:922.000", "frame:1312.000", "frame:3127.000", "frame:3157.000", "frame:3487.000"]`
- 新理由：The construction frames show only a subset of the very large selected tool inventory, such as gloves, bricks, string/line and striking/digging tools; they cannot establish the exhaustive list of all claimed unique tools.
- 新证据引用：`["image:frames_001-009@7-3487s"]`
- 具体分歧：两轮都只看到二十五项工具清单中的一部分；旧审把已见子集记作部分支持，新审因无法验证完整清单而记作无支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（9张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。

## 31. E0892: `unsupported` → `partially_supported`

- 旧理由：The map and three frames show food preparation and jar/ingredient handling, but do not exhaustively cover the video well enough to establish the negative claim that no paper was cut and no rice measured. Non-mention cannot prove both events absent.
- 旧证据引用：`["frame:855.000", "frame:900.000", "frame:1080.000", "map:C06", "map:C07"]`
- 新理由：The available map and frames do not show paper cutting or an explicit rice-measuring event, but absence of those events across the full video cannot be established from the sampled evidence alone.
- 新证据引用：`["image:frames_001-003@855-1080s", "map:C07-C08"]`
- 具体分歧：两轮都认为稀疏证据不能证明全程没有剪纸和量米；旧审记无支持，新审把“现有证据未出现两事”保留为部分支持。
- 源材料：问题/选项/预测/reason一致=True；地图字节一致=True；图片SHA/顺序/resolved时间一致=True（3张）；修正包时间文字有效=True。
- 实际展示：未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。
- 诊断归类：`label_boundary_difference`；该归类不裁决哪轮正确。
