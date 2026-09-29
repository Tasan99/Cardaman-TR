"""Shared beverage domain vs pack-specific classes (Cardaman TR steps 4–7)."""
import unittest

from regchain.tr.packs import DATA, Registry

BUILTIN = Registry.load(DATA)
SHARED = (
    'PRODUCTION', 'BOTTLING', 'PACKAGING', 'WAREHOUSING', 'DISTRIBUTION', 'WHOLESALE', 'RETAIL_SALE',
    'HORECA_SUPPLY', 'ECOMMERCE_SALE', 'ADVERTISING', 'SPONSORSHIP', 'PROMOTION', 'IMPORT', 'EXPORT',
)
SHARED_FACILITIES = ('MANUFACTURING_PLANT', 'BOTTLING_PLANT', 'WAREHOUSE', 'DISTRIBUTION_CENTER', 'OFFICE', 'LABORATORY')


class SharedDomainTests(unittest.TestCase):
    def test_shared_activities_and_facilities_live_on_the_common_module(self):
        common = BUILTIN.pack('TR_FOOD_BEVERAGE_COMMON')
        for item in SHARED:
            self.assertIn(item, common.activity_classes, item)
        for item in SHARED_FACILITIES:
            self.assertIn(item, common.facility_classes, item)

    def test_both_packs_include_the_common_module_instead_of_owning_shared_law(self):
        for pack_id in ('BEVERAGE_ALCOHOL_TR', 'BEVERAGE_NON_ALCOHOL_TR'):
            pack = BUILTIN.pack(pack_id)
            self.assertIn('TR_FOOD_BEVERAGE_COMMON', pack.includes)
            self.assertIn('TR:YONETMELIK:AMBALAJ_ATIKLARI', BUILTIN.regulation_ids(pack_id))
            self.assertEqual(BUILTIN.owner('TR:YONETMELIK:AMBALAJ_ATIKLARI'), 'TR_FOOD_BEVERAGE_COMMON')

    def test_alcohol_specific_classes_are_not_copied_into_the_non_alcohol_pack(self):
        alcohol = BUILTIN.pack('BEVERAGE_ALCOHOL_TR')
        soft = BUILTIN.pack('BEVERAGE_NON_ALCOHOL_TR')
        for item in ('BEER', 'LOW_ALCOHOL_BEER', 'WINE', 'SPIRIT', 'BREWING', 'BREWERY'):
            owned = alcohol.product_classes + alcohol.activity_classes + alcohol.facility_classes
            self.assertIn(item, owned, item)
            self.assertNotIn(item, soft.product_classes + soft.activity_classes + soft.facility_classes, item)

    def test_non_alcohol_specific_classes_are_not_copied_into_the_alcohol_pack(self):
        alcohol = BUILTIN.pack('BEVERAGE_ALCOHOL_TR')
        soft = BUILTIN.pack('BEVERAGE_NON_ALCOHOL_TR')
        for item in ('ENERGY_DRINK', 'PACKAGED_WATER', 'FRUIT_DRINK', 'SYRUP', 'CONCENTRATE', 'WATER_BOTTLING_PLANT'):
            owned = soft.product_classes + soft.activity_classes + soft.facility_classes
            self.assertIn(item, owned, item)
            self.assertNotIn(item, alcohol.product_classes + alcohol.activity_classes + alcohol.facility_classes, item)


if __name__ == '__main__':
    unittest.main()
